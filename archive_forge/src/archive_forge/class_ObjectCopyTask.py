from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class ObjectCopyTask(CopyTask):
    """Parent task that handles common attributes for object copy tasks."""

    def __init__(self, source_resource, destination_resource, posix_to_set=None, print_created_message=False, print_source_version=False, user_request_args=None, verbose=False):
        """Initializes task.

    Args:
      source_resource (resource_reference.Resource): See parent class.
      destination_resource (resource_reference.Resource): See parent class.
      posix_to_set (PosixAttributes|None): POSIX info set as custom cloud
        metadata on target.
      print_created_message (bool): See parent class.
      print_source_version (bool): See parent class.
      user_request_args (UserRequestArgs|None): See parent class.
      verbose (bool): Print a "copying" status message on initialization.
    """
        self._posix_to_set = posix_to_set
        self._print_source_version = print_source_version
        super(ObjectCopyTask, self).__init__(source_resource, destination_resource, print_created_message, print_source_version, user_request_args, verbose)