from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import path_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_factory
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _raise_if_download_destination_ends_with_delimiter_and_does_not_exist(self):
    if isinstance(self._raw_destination.storage_url, storage_url.FileUrl):
        destination_path = self._raw_destination.storage_url.object_name
        if destination_path.endswith(self._raw_destination.storage_url.delimiter) and (not self._raw_destination.storage_url.isdir()):
            raise errors.InvalidUrlError('Destination URL must name an existing directory if it ends with a delimiter. Provided: {}.'.format(destination_path))