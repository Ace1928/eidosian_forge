import boto
import os
import sys
import textwrap
from boto.s3.deletemarker import DeleteMarker
from boto.exception import BotoClientError
from boto.exception import InvalidUriError
class FileStorageUri(StorageUri):
    """
    StorageUri subclass that handles files in the local file system.
    Callers should instantiate this class by calling boto.storage_uri().

    See file/README about how we map StorageUri operations onto a file system.
    """
    delim = os.sep

    def __init__(self, object_name, debug, is_stream=False):
        """Instantiate a FileStorageUri from a path name.

        @type object_name: string
        @param object_name: object name
        @type debug: boolean
        @param debug: whether to enable debugging on this StorageUri

        After instantiation the components are available in the following
        fields: uri, scheme, bucket_name (always blank for this "anonymous"
        bucket), object_name.
        """
        self.scheme = 'file'
        self.bucket_name = ''
        self.object_name = object_name
        self.uri = 'file://' + object_name
        self.debug = debug
        self.stream = is_stream

    def clone_replace_name(self, new_name):
        """Instantiate a FileStorageUri from the current FileStorageUri,
        but replacing the object_name.

        @type new_name: string
        @param new_name: new object name
        """
        return FileStorageUri(new_name, self.debug, self.stream)

    def is_file_uri(self):
        """Returns True if this URI names a file or directory."""
        return True

    def is_cloud_uri(self):
        """Returns True if this URI names a bucket or object."""
        return False

    def names_container(self):
        """Returns True if this URI names a directory or bucket."""
        return self.names_directory()

    def names_singleton(self):
        """Returns True if this URI names a file (or stream) or object."""
        return not self.names_container()

    def names_directory(self):
        """Returns True if this URI names a directory."""
        if self.stream:
            return False
        return os.path.isdir(self.object_name)

    def names_provider(self):
        """Returns True if this URI names a provider."""
        return False

    def names_bucket(self):
        """Returns True if this URI names a bucket."""
        return False

    def names_file(self):
        """Returns True if this URI names a file."""
        return self.names_singleton()

    def names_object(self):
        """Returns True if this URI names an object."""
        return False

    def is_stream(self):
        """Returns True if this URI represents input/output stream.
        """
        return bool(self.stream)

    def close(self):
        """Closes the underlying file.
        """
        self.get_key().close()

    def exists(self, _headers_not_used=None):
        """Returns True if the file exists or False if it doesn't"""
        return os.path.exists(self.object_name)