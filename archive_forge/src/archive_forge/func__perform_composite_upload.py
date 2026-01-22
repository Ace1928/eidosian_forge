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
def _perform_composite_upload(self, api_client, component_count, size, source_path, task_status_queue, temporary_paths_to_clean_up):
    tracker_file_path = tracker_file_util.get_tracker_file_path(self._destination_resource.storage_url, tracker_file_util.TrackerFileType.PARALLEL_UPLOAD, source_url=self._source_resource.storage_url)
    tracker_data = tracker_file_util.read_composite_upload_tracker_file(tracker_file_path)
    if tracker_data:
        random_prefix = tracker_data.random_prefix
    else:
        random_prefix = _get_random_prefix()
    component_offsets_and_lengths = copy_component_util.get_component_offsets_and_lengths(size, component_count)
    temporary_component_resources = []
    for i in range(len(component_offsets_and_lengths)):
        temporary_component_resource = copy_component_util.get_temporary_component_resource(self._source_resource, self._destination_resource, random_prefix, i)
        temporary_component_resources.append(temporary_component_resource)
        component_name_length = len(temporary_component_resource.storage_url.object_name.encode())
        if component_name_length > api_client.MAX_OBJECT_NAME_LENGTH:
            log.warning('Performing a non-composite upload for {}, as a temporary component resource would have a name of length {}. This is longer than the maximum object name length supported by this API: {} UTF-8 encoded bytes. You may be able to change the storage/parallel_composite_upload_prefix config option to perform a composite upload with this object.'.format(self._source_resource.storage_url, component_name_length, api_client.MAX_OBJECT_NAME_LENGTH))
            return self._perform_single_transfer(size, source_path, task_status_queue, temporary_paths_to_clean_up)
    file_part_upload_tasks = []
    for i, (offset, length) in enumerate(component_offsets_and_lengths):
        upload_task = file_part_upload_task.FilePartUploadTask(self._source_resource, temporary_component_resources[i], source_path, offset, length, component_number=i, total_components=len(component_offsets_and_lengths), user_request_args=self._user_request_args)
        file_part_upload_tasks.append(upload_task)
    finalize_upload_task = finalize_composite_upload_task.FinalizeCompositeUploadTask(expected_component_count=len(file_part_upload_tasks), source_resource=self._source_resource, destination_resource=self._destination_resource, delete_source=self._delete_source, posix_to_set=self._posix_to_set, print_created_message=self._print_created_message, random_prefix=random_prefix, temporary_paths_to_clean_up=temporary_paths_to_clean_up, user_request_args=self._user_request_args)
    tracker_file_util.write_composite_upload_tracker_file(tracker_file_path, random_prefix)
    return task.Output(additional_task_iterators=[file_part_upload_tasks, [finalize_upload_task]], messages=None)