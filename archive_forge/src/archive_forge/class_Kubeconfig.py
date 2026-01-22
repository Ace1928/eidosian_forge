from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import os
from typing import Any
from googlecloudsdk.api_lib.container import kubeconfig as container_kubeconfig
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
class Kubeconfig(object):
    """Interface for interacting with a kubeconfig file."""

    def __init__(self, raw_data: dict[str, Any], filename: str):
        self._filename = filename
        self._data = raw_data
        self.clusters = {}
        self.users = {}
        self.contexts = {}
        for cluster in self._data['clusters']:
            self.clusters[cluster['name']] = cluster
        for user in self._data['users']:
            self.users[user['name']] = user
        for context in self._data['contexts']:
            self.contexts[context['name']] = context

    @property
    def current_context(self):
        return self._data['current-context']

    @property
    def filename(self):
        return self._filename

    def Clear(self, key):
        self.contexts.pop(key, None)
        self.clusters.pop(key, None)
        self.users.pop(key, None)
        if self._data.get('current-context') == key:
            self._data['current-context'] = ''

    def SaveToFile(self):
        """Save kubeconfig to file.

    Raises:
      Error: don't have the permission to open kubeconfig file.
    """
        self._data['clusters'] = list(self.clusters.values())
        self._data['users'] = list(self.users.values())
        self._data['contexts'] = list(self.contexts.values())
        _ = yaml.dump(self._data, None)
        with file_utils.FileWriter(self._filename, private=True) as fp:
            yaml.dump(self._data, fp)
        dirname = os.path.dirname(self._filename)
        gke_gcloud_auth_plugin_file_path = os.path.join(dirname, GKE_GCLOUD_AUTH_PLUGIN_CACHE_FILE_NAME)
        if os.path.exists(gke_gcloud_auth_plugin_file_path):
            file_utils.WriteFileAtomically(gke_gcloud_auth_plugin_file_path, '')

    def SetCurrentContext(self, context):
        self._data['current-context'] = context

    @classmethod
    def _Validate(cls, data):
        """Make sure we have the main fields of a kubeconfig."""
        if not data:
            raise Error('empty file')
        try:
            for key in ('clusters', 'users', 'contexts'):
                if not isinstance(data[key], list):
                    raise Error('invalid type for {0}: {1}'.format(data[key], type(data[key])))
        except KeyError as error:
            raise Error('expected key {0} not found'.format(error))

    @classmethod
    def LoadFromFile(cls, filename):
        try:
            data = yaml.load_path(filename)
        except yaml.Error as error:
            raise Error('unable to load kubeconfig for {0}: {1}'.format(filename, error.inner_error))
        cls._Validate(data)
        return cls(data, filename)

    @classmethod
    def LoadFromBytes(cls, raw_data: bytes, path: str=None) -> Kubeconfig:
        """Parse a YAML kubeconfig.

    Args:
      raw_data: The YAML data to parse
      path: The path to associate with the data. Defaults to calling
        `Kubeconfig.DefaultPath()`.

    Returns:
      A `Kubeconfig` instance.

    Raises:
      Error: The data is not valid YAML.
    """
        try:
            data = yaml.load(raw_data)
        except yaml.Error as error:
            raise Error(f'unable to parse kubeconfig bytes: {error.inner_error}')
        cls._Validate(data)
        if not path:
            path = cls.DefaultPath()
        return cls(data, path)

    @classmethod
    def LoadOrCreate(cls, filename):
        """Read in the kubeconfig, and if it doesn't exist create one there."""
        try:
            return cls.LoadFromFile(filename)
        except (Error, IOError) as error:
            log.debug('unable to load default kubeconfig: {0}; recreating {1}'.format(error, filename))
            file_utils.MakeDir(os.path.dirname(filename))
            kubeconfig = cls(EmptyKubeconfig(), filename)
            kubeconfig.SaveToFile()
            return kubeconfig

    @classmethod
    def Default(cls):
        return cls.LoadOrCreate(Kubeconfig.DefaultPath())

    @staticmethod
    def DefaultPath():
        """Return default path for kubeconfig file."""
        kubeconfig = encoding.GetEncodedValue(os.environ, 'KUBECONFIG')
        if kubeconfig:
            kubeconfig = kubeconfig.split(os.pathsep)[0]
            return os.path.abspath(kubeconfig)
        home_dir = encoding.GetEncodedValue(os.environ, 'HOME')
        if not home_dir and platforms.OperatingSystem.IsWindows():
            home_drive = encoding.GetEncodedValue(os.environ, 'HOMEDRIVE')
            home_path = encoding.GetEncodedValue(os.environ, 'HOMEPATH')
            if home_drive and home_path:
                home_dir = os.path.join(home_drive, home_path)
            if not home_dir:
                home_dir = encoding.GetEncodedValue(os.environ, 'USERPROFILE')
        if not home_dir:
            raise MissingEnvVarError('environment variable {vars} or KUBECONFIG must be set to store credentials for kubectl'.format(vars='HOMEDRIVE/HOMEPATH, USERPROFILE, HOME,' if platforms.OperatingSystem.IsWindows() else 'HOME'))
        return os.path.join(home_dir, '.kube', 'config')

    def Merge(self, kubeconfig: Kubeconfig, overwrite: bool=False) -> None:
        """Merge another kubeconfig into self.

    By default, in case of overlapping keys, the value in self is kept and the
    value in the other kubeconfig is lost.

    Args:
      kubeconfig: a Kubeconfig instance
      overwrite: whether to overwrite overlapping keys in self with data from
        the other kubeconfig.
    """
        left, right = (self, kubeconfig)
        if overwrite:
            left, right = (right, left)
        self.SetCurrentContext(left.current_context or right.current_context)
        self.clusters = dict(list(right.clusters.items()) + list(left.clusters.items()))
        self.users = dict(list(right.users.items()) + list(left.users.items()))
        self.contexts = dict(list(right.contexts.items()) + list(left.contexts.items()))