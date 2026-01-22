from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import tracker_file_util
from googlecloudsdk.command_lib.storage.tasks import compose_objects_task
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_util
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import delete_temporary_components_task
Initializes task.

    Args:
      expected_component_count (int): Number of temporary components expected.
      source_resource (resource_reference.FileObjectResource): The local
        uploaded file.
      destination_resource (resource_reference.UnknownResource): Metadata for
        the final composite object.
      delete_source (bool): If copy completes successfully, delete the source
        object afterwards.
      posix_to_set (PosixAttributes|None): See parent class.
      print_created_message (bool): See parent class.
      random_prefix (str): Random id added to component names.
      temporary_paths_to_clean_up (str): Paths to remove after the composite
        upload completes. This may include a temporary gzipped version of the
        source, or symlink placeholders.
      user_request_args (UserRequestArgs|None): See parent class.
    