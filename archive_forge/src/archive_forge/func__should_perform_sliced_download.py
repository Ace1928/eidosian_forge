from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import posix_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import symlink_util
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_component_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.command_lib.storage.tasks.cp import file_part_download_task
from googlecloudsdk.command_lib.storage.tasks.cp import finalize_sliced_download_task
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import scaled_integer
def _should_perform_sliced_download(source_resource, destination_resource):
    """Returns True if conditions are right for a sliced download."""
    if destination_resource.storage_url.is_stream:
        return False
    if not source_resource.crc32c_hash and properties.VALUES.storage.check_hashes.Get() != properties.CheckHashes.NEVER.value:
        return False
    threshold = scaled_integer.ParseInteger(properties.VALUES.storage.sliced_object_download_threshold.Get())
    component_size = scaled_integer.ParseInteger(properties.VALUES.storage.sliced_object_download_component_size.Get())
    api_capabilities = api_factory.get_capabilities(source_resource.storage_url.scheme)
    return source_resource.size and threshold != 0 and (source_resource.size > threshold) and component_size and (cloud_api.Capability.SLICED_DOWNLOAD in api_capabilities) and task_util.should_use_parallelism()