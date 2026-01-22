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
def _get_copy_destination(self, raw_destination, source):
    """Returns the final destination StorageUrl instance."""
    completion_is_necessary = _destination_is_container(raw_destination) or (self._multiple_sources and (not _resource_is_stream(raw_destination))) or source.resource.storage_url.versionless_url_string != source.expanded_url.versionless_url_string
    if completion_is_necessary:
        if isinstance(source.expanded_url, storage_url.FileUrl) and source.expanded_url.is_stdio:
            raise errors.Error('Destination object name needed when source is stdin.')
        destination_resource = self._complete_destination(raw_destination, source)
    else:
        destination_resource = raw_destination
    sanitized_destination_resource = path_util.sanitize_file_resource_for_windows(destination_resource)
    return sanitized_destination_resource