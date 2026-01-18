from pathlib import Path
from typing import (
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import DeleteMixin, GetMixin, ListMixin, ObjectDeleteMixin
@cli.register_custom_action('GenericPackageManager', ('package_name', 'package_version', 'file_name', 'path'))
@exc.on_http_error(exc.GitlabUploadError)
def upload(self, package_name: str, package_version: str, file_name: str, path: Optional[Union[str, Path]]=None, select: Optional[str]=None, data: Optional[Union[bytes, BinaryIO]]=None, **kwargs: Any) -> GenericPackage:
    """Upload a file as a generic package.

        Args:
            package_name: The package name. Must follow generic package
                                name regex rules
            package_version: The package version. Must follow semantic
                                version regex rules
            file_name: The name of the file as uploaded in the registry
            path: The path to a local file to upload
            select: GitLab API accepts a value of 'package_file'

        Raises:
            GitlabConnectionError: If the server cannot be reached
            GitlabUploadError: If the file upload fails
            GitlabUploadError: If ``path`` cannot be read
            GitlabUploadError: If both ``path`` and ``data`` are passed

        Returns:
            An object storing the metadata of the uploaded package.

        https://docs.gitlab.com/ee/user/packages/generic_packages/
        """
    if path is None and data is None:
        raise exc.GitlabUploadError('No file contents or path specified')
    if path is not None and data is not None:
        raise exc.GitlabUploadError('File contents and file path specified')
    file_data: Optional[Union[bytes, BinaryIO]] = data
    if not file_data:
        if TYPE_CHECKING:
            assert path is not None
        try:
            with open(path, 'rb') as f:
                file_data = f.read()
        except OSError as e:
            raise exc.GitlabUploadError(f'Failed to read package file {path}') from e
    url = f'{self._computed_path}/{package_name}/{package_version}/{file_name}'
    query_data = {} if select is None else {'select': select}
    server_data = self.gitlab.http_put(url, query_data=query_data, post_data=file_data, raw=True, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(server_data, dict)
    attrs = {'package_name': package_name, 'package_version': package_version, 'file_name': file_name, 'path': path}
    attrs.update(server_data)
    return self._obj_cls(self, attrs=attrs)