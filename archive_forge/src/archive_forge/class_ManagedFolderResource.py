from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_util
class ManagedFolderResource(PrefixResource):
    """Class representing a managed folder."""
    TYPE_STRING = 'managed_folder'

    def __init__(self, storage_url_object, create_time=None, metadata=None, metageneration=None, update_time=None):
        super(ManagedFolderResource, self).__init__(storage_url_object, storage_url_object.object_name)
        self.create_time = create_time
        self.metadata = metadata
        self.metageneration = metageneration
        self.update_time = update_time

    @property
    def bucket(self):
        return self.storage_url.bucket_name

    @property
    def is_symlink(self):
        return False

    @property
    def name(self):
        return self.storage_url.object_name

    def __eq__(self, other):
        return super(ManagedFolderResource, self).__eq__(other) and self.storage_url == other.storage_url and (self.create_time == other.create_time) and (self.metadata == other.metadata) and (self.metageneration == other.metageneration) and (self.update_time == other.update_time)