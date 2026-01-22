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
def _update_workload_estimation(self, resource):
    """Updates total_file_count and total_size.

    Args:
      resource (resource_reference.Resource): Any type of resource. Parse to
        help estimate total workload.
    """
    if self._total_file_count == -1 or self._total_size == -1:
        return
    try:
        if resource.is_container():
            return
        size = resource.size
        if isinstance(resource, resource_reference.FileObjectResource):
            self._has_local_source = True
        elif isinstance(resource, resource_reference.ObjectResource):
            self._has_cloud_source = True
        else:
            raise errors.ValueCannotBeDeterminedError
    except (OSError, errors.ValueCannotBeDeterminedError):
        if not _resource_is_stream(resource):
            log.error('Could not get size of resource {}.'.format(resource))
        self._total_file_count = -1
        self._total_size = -1
    else:
        self._total_file_count += 1
        self._total_size += size or 0