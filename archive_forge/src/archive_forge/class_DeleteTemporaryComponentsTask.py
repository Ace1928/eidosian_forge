from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks.cp import copy_component_util
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
class DeleteTemporaryComponentsTask(task.Task):
    """Deletes temporary components and tracker files after a composite upload."""

    def __init__(self, source_resource, destination_resource, random_prefix):
        """Initializes a task instance.

    Args:
      source_resource (resource_reference.FileObjectResource): The local,
          uploaded file.
      destination_resource (resource_reference.UnknownResource): The final
          composite object's metadata.
      random_prefix (str): ID added to temporary component names.
    """
        super(DeleteTemporaryComponentsTask, self).__init__()
        self._source_resource = source_resource
        self._destination_resource = destination_resource
        self._random_prefix = random_prefix

    def execute(self, task_status_queue=None):
        """Deletes temporary components and associated tracker files.

    Args:
      task_status_queue: See base class.

    Returns:
      A task.Output with tasks for deleting temporary components.
    """
        del task_status_queue
        component_tracker_path_prefix = tracker_file_util.get_tracker_file_path(copy_component_util.get_temporary_component_resource(self._source_resource, self._destination_resource, self._random_prefix, component_id='').storage_url, tracker_file_util.TrackerFileType.UPLOAD, component_number='')
        component_tracker_paths = glob.iglob(component_tracker_path_prefix + '*')
        component_urls = []
        found_permissions_error = permissions_error = None
        for component_tracker_path in component_tracker_paths:
            tracker_data = tracker_file_util.read_resumable_upload_tracker_file(component_tracker_path)
            if tracker_data.complete:
                _, _, component_number = component_tracker_path.rpartition('_')
                component_url = copy_component_util.get_temporary_component_resource(self._source_resource, self._destination_resource, self._random_prefix, component_id=component_number).storage_url
                if found_permissions_error is None:
                    permissions_error = _try_delete_and_return_permissions_error(component_url)
                    found_permissions_error = permissions_error is not None
                    if found_permissions_error:
                        component_urls.append(component_url)
                else:
                    component_urls.append(component_url)
            os.remove(component_tracker_path)
        if permissions_error:
            log.error('Parallel composite upload failed: Permissions error detected while attempting to delete object component.\n\nTo disable parallel composite uploads, run:\ngcloud config set storage/parallel_composite_upload_enabled False\n\nTo delete the temporary objects left over by this command, switch to an account with appropriate permissions and run:\ngcloud storage rm {}'.format(' '.join([url.url_string for url in component_urls])))
            raise command_errors.FatalError(permissions_error)
        if component_urls:
            additional_task_iterators = [[delete_task.DeleteObjectTask(url, verbose=False) for url in component_urls]]
            return task.Output(additional_task_iterators=additional_task_iterators, messages=None)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._source_resource == other._source_resource and self._destination_resource == other._destination_resource and (self._random_prefix == other._random_prefix)