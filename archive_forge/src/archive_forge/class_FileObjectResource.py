from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
class FileObjectResource(Resource):
    """Wrapper for a filesystem file.

  Attributes:
    TYPE_STRING (str): String representing the resource's content type.
    size (int|None): Size of local file in bytes or None if pipe or stream.
    storage_url (StorageUrl): A StorageUrl object representing the resource.
    md5_hash (bytes): Base64-encoded digest of MD5 hash.
    is_symlink (bool|None): Whether this file is known to be a symlink.
  """
    TYPE_STRING = 'file_object'

    def __init__(self, storage_url_object, md5_hash=None, is_symlink=None):
        """Initializes resource. Args are a subset of attributes."""
        super(FileObjectResource, self).__init__(storage_url_object)
        self.md5_hash = md5_hash
        self._is_symlink = is_symlink

    def is_container(self):
        return False

    @property
    def size(self):
        """Returns file size or None if pipe or stream."""
        if self.storage_url.is_stream:
            return None
        return os.path.getsize(self.storage_url.object_name)

    @property
    def is_symlink(self):
        """Returns whether this file is a symlink."""
        if self._is_symlink is None:
            self._is_symlink = os.path.islink(self.storage_url.object_name)
        return self._is_symlink