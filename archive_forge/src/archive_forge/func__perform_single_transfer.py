from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import random
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.command_lib.storage import gzip_util
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import symlink_util
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_component_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import file_part_upload_task
from googlecloudsdk.command_lib.storage.tasks.cp import finalize_composite_upload_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _perform_single_transfer(self, size, source_path, task_status_queue, temporary_paths_to_clean_up):
    task_output = file_part_upload_task.FilePartUploadTask(self._source_resource, self._destination_resource, source_path, offset=0, length=size, posix_to_set=self._posix_to_set, user_request_args=self._user_request_args).execute(task_status_queue)
    result_resource = task_util.get_first_matching_message_payload(task_output.messages, task.Topic.CREATED_RESOURCE)
    if result_resource:
        self._print_created_message_if_requested(result_resource)
        if self._send_manifest_messages:
            manifest_util.send_success_message(task_status_queue, self._source_resource, self._destination_resource, md5_hash=result_resource.md5_hash)
    for path in temporary_paths_to_clean_up:
        os.remove(path)
    if self._delete_source:
        os.remove(self._source_resource.storage_url.object_name)