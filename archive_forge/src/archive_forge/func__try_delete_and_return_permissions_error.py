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
def _try_delete_and_return_permissions_error(component_url):
    """Attempts deleting component and returns any permissions errors."""
    try:
        delete_task.DeleteObjectTask(component_url, verbose=False).execute()
    except api_errors.CloudApiError as e:
        status = getattr(e, 'status_code', None)
        if status == 403:
            return e
        raise