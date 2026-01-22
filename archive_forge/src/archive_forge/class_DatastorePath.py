import logging
import posixpath
import random
import re
import http.client as httplib
import urllib.parse as urlparse
from oslo_vmware._i18n import _
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import vim_util
class DatastorePath(object):
    """Class for representing a directory or file path in a vSphere datatore.

    This provides various helper methods to access components and useful
    variants of the datastore path.

    Example usage:

    DatastorePath("datastore1", "_base/foo", "foo.vmdk") creates an
    object that describes the "[datastore1] _base/foo/foo.vmdk" datastore
    file path to a virtual disk.

    Note:

    - Datastore path representations always uses forward slash as separator
      (hence the use of the posixpath module).
    - Datastore names are enclosed in square brackets.
    - Path part of datastore path is relative to the root directory
      of the datastore, and is always separated from the [ds_name] part with
      a single space.
    """

    def __init__(self, datastore_name, *paths):
        if datastore_name is None or datastore_name == '':
            raise ValueError(_('Datastore name cannot be empty'))
        self._datastore_name = datastore_name
        self._rel_path = ''
        if paths:
            if None in paths:
                raise ValueError(_('Path component cannot be None'))
            self._rel_path = posixpath.join(*paths)

    def __str__(self):
        """Full datastore path to the file or directory."""
        if self._rel_path != '':
            return '[%s] %s' % (self._datastore_name, self.rel_path)
        return '[%s]' % self._datastore_name

    @property
    def datastore(self):
        return self._datastore_name

    @property
    def parent(self):
        return DatastorePath(self.datastore, posixpath.dirname(self._rel_path))

    @property
    def basename(self):
        return posixpath.basename(self._rel_path)

    @property
    def dirname(self):
        return posixpath.dirname(self._rel_path)

    @property
    def rel_path(self):
        return self._rel_path

    def join(self, *paths):
        """Join one or more path components intelligently into a datastore path.

        If any component is an absolute path, all previous components are
        thrown away, and joining continues. The return value is the
        concatenation of the paths with exactly one slash ('/') inserted
        between components, unless p is empty.

        :return: A datastore path
        """
        if paths:
            if None in paths:
                raise ValueError(_('Path component cannot be None'))
            return DatastorePath(self.datastore, self._rel_path, *paths)
        return self

    def __eq__(self, other):
        return isinstance(other, DatastorePath) and self._datastore_name == other._datastore_name and (self._rel_path == other._rel_path)

    @classmethod
    def parse(cls, datastore_path):
        """Constructs a DatastorePath object given a datastore path string."""
        if not datastore_path:
            raise ValueError(_('Datastore path cannot be empty'))
        spl = datastore_path.split('[', 1)[1].split(']', 1)
        path = ''
        if len(spl) == 1:
            datastore_name = spl[0]
        else:
            datastore_name, path = spl
        return cls(datastore_name, path.strip())