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
def _perform_resumable_download(self, request_config, progress_callback, digesters):
    """Resume or start download that can be resumabled."""
    copy_component_util.create_file_if_needed(self._source_resource, self._destination_resource)
    destination_url = self._destination_resource.storage_url
    first_null_byte = _get_first_null_byte_index(destination_url, self._offset, self._length)
    _, found_tracker_file = tracker_file_util.read_or_create_download_tracker_file(self._source_resource, destination_url)
    start_byte = first_null_byte if found_tracker_file else 0
    end_byte = self._source_resource.size - 1
    if start_byte:
        write_mode = files.BinaryFileWriterMode.MODIFY
        self._catch_up_digesters(digesters, start_byte=0, end_byte=start_byte)
        log.status.Print('Resuming download for {}'.format(self._source_resource))
    else:
        write_mode = files.BinaryFileWriterMode.TRUNCATE
    return self._perform_download(request_config, progress_callback, self._disable_in_flight_decompression(True), cloud_api.DownloadStrategy.RESUMABLE, start_byte, end_byte, write_mode, digesters)