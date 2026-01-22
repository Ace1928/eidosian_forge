import os
import posixpath
import re
import urllib.parse
from typing import List
import requests
from mlflow.azure.client import patch_adls_file_upload, patch_adls_flush, put_adls_file_creation
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.store.artifact.cloud_artifact_repo import (
class AzureDataLakeArtifactRepository(CloudArtifactRepository):
    """
    Stores artifacts on Azure Data Lake Storage Gen2.

    This repository is used with URIs of the form
    ``abfs[s]://file_system@account_name.dfs.core.windows.net/<path>/<path>``.

    Args
        credential: Azure credential (see options in https://learn.microsoft.com/en-us/python/api/azure-core/azure.core.credentials?view=azure-python)
            to use to authenticate to storage
    """

    def __init__(self, artifact_uri, credential):
        super().__init__(artifact_uri)
        _DEFAULT_TIMEOUT = 600
        self.credential = credential
        self.write_timeout = MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT.get() or _DEFAULT_TIMEOUT
        filesystem, account_name, domain_suffix, path = _parse_abfss_uri(artifact_uri)
        account_url = f'https://{account_name}.{domain_suffix}'
        data_lake_client = _get_data_lake_client(account_url=account_url, credential=credential)
        self.fs_client = data_lake_client.get_file_system_client(filesystem)
        self.domain_suffix = domain_suffix
        self.base_data_lake_directory = path
        self.account_name = account_name
        self.container = filesystem

    def log_artifact(self, local_file, artifact_path=None):
        dest_path = self.base_data_lake_directory
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        local_file_path = os.path.abspath(local_file)
        file_name = os.path.basename(local_file_path)
        dir_client = self.fs_client.get_directory_client(dest_path)
        file_client = dir_client.get_file_client(file_name)
        if os.path.getsize(local_file_path) == 0:
            file_client.create_file()
        else:
            with open(local_file_path, 'rb') as file:
                file_client.upload_data(data=file, overwrite=True)

    def list_artifacts(self, path=None):
        directory_to_list = self.base_data_lake_directory
        if path:
            directory_to_list = posixpath.join(directory_to_list, path)
        infos = []
        for result in self.fs_client.get_paths(path=directory_to_list, recursive=False):
            if directory_to_list == result.name:
                continue
            if result.is_directory:
                subdir = posixpath.relpath(path=result.name, start=self.base_data_lake_directory)
                if subdir.endswith('/'):
                    subdir = subdir[:-1]
                infos.append(FileInfo(subdir, is_dir=True, file_size=None))
            else:
                file_name = posixpath.relpath(path=result.name, start=self.base_data_lake_directory)
                infos.append(FileInfo(file_name, is_dir=False, file_size=result.content_length))
        rel_path = directory_to_list[len(self.base_data_lake_directory) + 1:]
        if len(infos) == 1 and (not infos[0].is_dir) and (infos[0].path == rel_path):
            return []
        return sorted(infos, key=lambda f: f.path)

    def _download_from_cloud(self, remote_file_path, local_path):
        remote_full_path = posixpath.join(self.base_data_lake_directory, remote_file_path)
        base_dir = posixpath.dirname(remote_full_path)
        dir_client = self.fs_client.get_directory_client(base_dir)
        filename = posixpath.basename(remote_full_path)
        file_client = dir_client.get_file_client(filename)
        with open(local_path, 'wb') as file:
            file_client.download_file().readinto(file)

    def delete_artifacts(self, artifact_path=None):
        raise NotImplementedError('This artifact repository does not support deleting artifacts')

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
        if MLFLOW_ENABLE_MULTIPART_UPLOAD.get() and os.path.getsize(src_file_path) > MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get():
            self._multipart_upload(cloud_credential_info, src_file_path, artifact_file_path)
        else:
            artifact_subdir = posixpath.dirname(artifact_file_path)
            self.log_artifact(src_file_path, artifact_subdir)

    def _retryable_adls_function(self, func, artifact_file_path, **kwargs):
        try:
            func(**kwargs)
        except requests.HTTPError as e:
            if e.response.status_code in [403]:
                new_credentials = self._get_write_credential_infos([artifact_file_path])[0]
                kwargs['sas_url'] = new_credentials.signed_uri
                func(**kwargs)
            else:
                raise e

    def _multipart_upload(self, credentials, src_file_path, artifact_file_path):
        """
        Uploads a file to a given Azure storage location using the ADLS gen2 API.
        """
        try:
            headers = self._extract_headers_from_credentials(credentials.headers)
            self._retryable_adls_function(func=put_adls_file_creation, artifact_file_path=artifact_file_path, sas_url=credentials.signed_uri, headers=headers)
            futures = {}
            file_size = os.path.getsize(src_file_path)
            num_chunks = _compute_num_chunks(src_file_path, MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())
            use_single_part_upload = num_chunks == 1
            for index in range(num_chunks):
                start_byte = index * MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
                future = self.chunk_thread_pool.submit(self._retryable_adls_function, func=patch_adls_file_upload, artifact_file_path=artifact_file_path, sas_url=credentials.signed_uri, local_file=src_file_path, start_byte=start_byte, size=MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get(), position=start_byte, headers=headers, is_single=use_single_part_upload)
                futures[future] = index
            _, errors = _complete_futures(futures, src_file_path)
            if errors:
                raise MlflowException(f'Failed to upload at least one part of {artifact_file_path}. Errors: {errors}')
            if not use_single_part_upload:
                self._retryable_adls_function(func=patch_adls_flush, artifact_file_path=artifact_file_path, sas_url=credentials.signed_uri, position=file_size, headers=headers)
        except Exception as err:
            raise MlflowException(err)

    def _get_presigned_uri(self, artifact_file_path):
        """
        Gets the presigned URL required to upload a file to or download a file from a given Azure
        storage location.

        Args:
            artifact_file_path: Path of the file relative to the artifact repository root.

        Returns:
            a string presigned URL.
        """
        sas_token = self.credential.signature
        return f'https://{self.account_name}.{self.domain_suffix}/{self.container}/{self.base_data_lake_directory}/{artifact_file_path}?{sas_token}'

    def _get_write_credential_infos(self, remote_file_paths) -> List[ArtifactCredentialInfo]:
        return [ArtifactCredentialInfo(signed_uri=self._get_presigned_uri(path)) for path in remote_file_paths]

    def _get_read_credential_infos(self, remote_file_paths) -> List[ArtifactCredentialInfo]:
        return [ArtifactCredentialInfo(signed_uri=self._get_presigned_uri(path)) for path in remote_file_paths]