import json
import os
import posixpath
import mlflow.utils.databricks_utils
from mlflow.entities import FileInfo
from mlflow.environment_variables import MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service import utils
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.rest_utils import RESOURCE_DOES_NOT_EXIST, http_request, http_request_safe
from mlflow.utils.string_utils import strip_prefix
from mlflow.utils.uri import (
class DbfsRestArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on DBFS using the DBFS REST API.

    This repository is used with URIs of the form ``dbfs:/<path>``. The repository can only be used
    together with the RestStore.
    """

    def __init__(self, artifact_uri):
        if not is_valid_dbfs_uri(artifact_uri):
            raise MlflowException(message='DBFS URI must be of the form dbfs:/<path> or ' + 'dbfs://profile@databricks/<path>', error_code=INVALID_PARAMETER_VALUE)
        super().__init__(remove_databricks_profile_info_from_artifact_uri(artifact_uri))
        databricks_profile_uri = get_databricks_profile_uri_from_artifact_uri(artifact_uri)
        if databricks_profile_uri:
            hostcreds_from_uri = get_databricks_host_creds(databricks_profile_uri)
            self.get_host_creds = lambda: hostcreds_from_uri
        else:
            self.get_host_creds = _get_host_creds_from_default_store()

    def _databricks_api_request(self, endpoint, method, **kwargs):
        host_creds = self.get_host_creds()
        return http_request_safe(host_creds=host_creds, endpoint=endpoint, method=method, **kwargs)

    def _dbfs_list_api(self, json):
        host_creds = self.get_host_creds()
        return http_request(host_creds=host_creds, endpoint=LIST_API_ENDPOINT, method='GET', params=json)

    def _dbfs_download(self, output_path, endpoint):
        with open(output_path, 'wb') as f:
            response = self._databricks_api_request(endpoint=endpoint, method='GET', stream=True)
            try:
                for content in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(content)
            finally:
                response.close()

    def _is_directory(self, artifact_path):
        dbfs_path = self._get_dbfs_path(artifact_path) if artifact_path else self._get_dbfs_path('')
        return self._dbfs_is_dir(dbfs_path)

    def _dbfs_is_dir(self, dbfs_path):
        response = self._databricks_api_request(endpoint=GET_STATUS_ENDPOINT, method='GET', params={'path': dbfs_path})
        json_response = json.loads(response.text)
        try:
            return json_response['is_dir']
        except KeyError:
            raise MlflowException(f'DBFS path {dbfs_path} does not exist')

    def _get_dbfs_path(self, artifact_path):
        return '/{}/{}'.format(strip_prefix(self.artifact_uri, 'dbfs:/'), strip_prefix(artifact_path, '/'))

    def _get_dbfs_endpoint(self, artifact_path):
        return f'/dbfs{self._get_dbfs_path(artifact_path)}'

    def log_artifact(self, local_file, artifact_path=None):
        basename = os.path.basename(local_file)
        if artifact_path:
            http_endpoint = self._get_dbfs_endpoint(posixpath.join(artifact_path, basename))
        else:
            http_endpoint = self._get_dbfs_endpoint(basename)
        if os.stat(local_file).st_size == 0:
            self._databricks_api_request(endpoint=http_endpoint, method='POST', data='', allow_redirects=False)
        else:
            with open(local_file, 'rb') as f:
                self._databricks_api_request(endpoint=http_endpoint, method='POST', data=f, allow_redirects=False)

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_path = artifact_path or ''
        for dirpath, _, filenames in os.walk(local_dir):
            artifact_subdir = artifact_path
            if dirpath != local_dir:
                rel_path = os.path.relpath(dirpath, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_subdir = posixpath.join(artifact_path, rel_path)
            for name in filenames:
                file_path = os.path.join(dirpath, name)
                self.log_artifact(file_path, artifact_subdir)

    def list_artifacts(self, path=None):
        dbfs_path = self._get_dbfs_path(path) if path else self._get_dbfs_path('')
        dbfs_list_json = {'path': dbfs_path}
        response = self._dbfs_list_api(dbfs_list_json)
        try:
            json_response = json.loads(response.text)
        except ValueError:
            raise MlflowException(f'API request to list files under DBFS path {dbfs_path} failed with status code {response.status_code}. Response body: {response.text}')
        infos = []
        artifact_prefix = strip_prefix(self.artifact_uri, 'dbfs:')
        if json_response.get('error_code', None) == RESOURCE_DOES_NOT_EXIST:
            return []
        dbfs_files = json_response.get('files', [])
        for dbfs_file in dbfs_files:
            stripped_path = strip_prefix(dbfs_file['path'], artifact_prefix + '/')
            if stripped_path == path:
                return []
            is_dir = dbfs_file['is_dir']
            artifact_size = None if is_dir else dbfs_file['file_size']
            infos.append(FileInfo(stripped_path, is_dir, artifact_size))
        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        self._dbfs_download(output_path=local_path, endpoint=self._get_dbfs_endpoint(remote_file_path))

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException('Not implemented yet')