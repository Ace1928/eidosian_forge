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
def _get_first_null_byte_index(destination_url, offset, length):
    """Checks to see how many bytes in range have already been downloaded.

  Args:
    destination_url (storage_url.FileUrl): Has path of file being downloaded.
    offset (int): For components, index to start reading bytes at.
    length (int): For components, where to stop reading bytes.

  Returns:
    Int byte count of size of partially-downloaded file. Returns 0 if file is
    an invalid size, empty, or non-existent.
  """
    if not destination_url.exists():
        return 0
    first_null_byte = offset
    end_of_range = offset + length
    with files.BinaryFileReader(destination_url.object_name) as file_reader:
        file_reader.seek(offset)
        while first_null_byte < end_of_range:
            data = file_reader.read(_READ_SIZE)
            if not data:
                break
            null_byte_index = data.find(NULL_BYTE)
            if null_byte_index != -1:
                first_null_byte += null_byte_index
                break
            first_null_byte += len(data)
    return first_null_byte