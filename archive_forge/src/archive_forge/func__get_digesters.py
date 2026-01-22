from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_component_util
from googlecloudsdk.command_lib.storage.tasks.cp import download_util
from googlecloudsdk.command_lib.storage.tasks.cp import file_part_task
from googlecloudsdk.command_lib.util import crc32c
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import hashing
def _get_digesters(component_number, resource):
    """Returns digesters dictionary for download hash validation.

  Note: The digester object is not picklable. It cannot be passed between
  tasks through the task graph.

  Args:
    component_number (int|None): Used to determine if downloading a slice in a
      sliced download, which uses CRC32C for hashing.
    resource (resource_reference.ObjectResource): For checking if object has
      known hash to validate against.

  Returns:
    Digesters dict.

  Raises:
    errors.Error: gcloud storage set to fail if performance-optimized digesters
      could not be created.
  """
    digesters = {}
    check_hashes = properties.VALUES.storage.check_hashes.Get()
    if check_hashes != properties.CheckHashes.NEVER.value:
        if component_number is None and resource.md5_hash:
            digesters[hash_util.HashAlgorithm.MD5] = hashing.get_md5()
        elif resource.crc32c_hash and (check_hashes == properties.CheckHashes.ALWAYS.value or fast_crc32c_util.check_if_will_use_fast_crc32c(install_if_missing=True)):
            digesters[hash_util.HashAlgorithm.CRC32C] = fast_crc32c_util.get_crc32c()
    if not digesters:
        log.warning('Found no hashes to validate download of object: %s. Integrity cannot be assured without hashes.', resource)
    return digesters