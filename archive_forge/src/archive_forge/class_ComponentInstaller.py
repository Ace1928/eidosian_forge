from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import tarfile
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import local_file_adapter
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import http_encoding
from googlecloudsdk.core.util import retry
import requests
import six
class ComponentInstaller(object):
    """A class to install Cloud SDK components of different source types."""
    DOWNLOAD_DIR_NAME = '.download'
    GCS_BROWSER_DL_URL = 'https://storage.cloud.google.com/'
    GCS_API_DL_URL = 'https://storage.googleapis.com/'

    def __init__(self, sdk_root, state_directory, snapshot):
        """Initializes an installer for components of different source types.

    Args:
      sdk_root:  str, The path to the root directory of all Cloud SDK files.
      state_directory: str, The path to the directory where the local state is
        stored.
      snapshot: snapshots.ComponentSnapshot, The snapshot that describes the
        component to install.
    """
        self.__sdk_root = sdk_root
        self.__state_directory = state_directory
        self.__download_directory = os.path.join(self.__state_directory, ComponentInstaller.DOWNLOAD_DIR_NAME)
        self.__snapshot = snapshot
        for d in [self.__download_directory]:
            if not os.path.isdir(d):
                file_utils.MakeDir(d)

    def Install(self, component_id, progress_callback=None, command_path='unknown'):
        """Installs the given component for whatever source type it has.

    Args:
      component_id: str, The component id from the snapshot to install.
      progress_callback: f(float), A function to call with the fraction of
        completeness.
      command_path: the command path to include in the User-Agent header if the
        URL is HTTP

    Returns:
      list of str, The files that were installed.

    Raises:
      UnsupportedSourceError: If the component data source is of an unknown
        type.
      URLFetchError: If the URL associated with the component data source
        cannot be fetched.
    """
        component = self.__snapshot.ComponentFromId(component_id)
        data = component.data
        if not data:
            return []
        if data.type == 'tar':
            return self._InstallTar(component, progress_callback=progress_callback, command_path=command_path)
        raise UnsupportedSourceError('tar is the only supported source format [{datatype}]'.format(datatype=data.type))

    def _InstallTar(self, component, progress_callback=None, command_path='unknown'):
        """Installer implementation for a component with source in a .tar.gz.

    Downloads the .tar for the component and extracts it.

    Args:
      component: schemas.Component, The component to install.
      progress_callback: f(float), A function to call with the fraction of
        completeness.
      command_path: the command path to include in the User-Agent header if the
        URL is HTTP

    Returns:
      list of str, The files that were installed or [] if nothing was installed.

    Raises:
      ValueError: If the source URL for the tar file is relative, but there is
        no location information associated with the snapshot we are installing
        from.
      URLFetchError: If there is a problem fetching the component's URL.
    """
        url = component.data.source
        if not url:
            return []
        if not re.search('^\\w+://', url):
            raise ValueError('Cannot install component [{0}] from a relative path because the base URL of the snapshot is not defined.'.format(component.id))
        try:
            return DownloadAndExtractTar(url, self.__download_directory, self.__sdk_root, progress_callback=progress_callback, command_path=command_path)
        except (URLFetchError, AuthenticationError) as e:
            raise ComponentDownloadFailedError(component.id, e)