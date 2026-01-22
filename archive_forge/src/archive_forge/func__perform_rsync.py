from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import cp_command_util
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import rsync_command_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks import get_sorted_list_file_task
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
def _perform_rsync(args, source_container, destination_container, perform_managed_folder_operations=False):
    """Creates and executes tasks for rsync commands.

  Args:
    args (parser_extensions.Namespace): Command line arguments.
    source_container (resource_reference.Resource): Location to find source
      resources.
    destination_container (resource_reference.Resource): Location for
      destination resources.
    perform_managed_folder_operations (bool): If True, generates manifest files
      and performs copy tasks for managed folders. Otherwise, does so for
      objects/files.

  Returns:
    Exit code (int).
  """
    user_request_args = user_request_args_factory.get_user_request_args_from_command_args(args, metadata_type=user_request_args_factory.MetadataType.OBJECT)
    source_list_path = rsync_command_util.get_hashed_list_file_path(source_container.storage_url.url_string, is_managed_folder_list=perform_managed_folder_operations)
    destination_list_path = rsync_command_util.get_hashed_list_file_path(destination_container.storage_url.url_string, is_managed_folder_list=perform_managed_folder_operations)
    task_status_queue = task_graph_executor.multiprocessing_context.Queue()
    operation_iterator = rsync_command_util.get_operation_iterator(user_request_args, source_list_path, source_container, destination_list_path, destination_container, compare_only_hashes=args.checksums_only, delete_unmatched_destination_objects=args.delete_unmatched_destination_objects, dry_run=args.dry_run, ignore_symlinks=args.ignore_symlinks, yield_managed_folder_operations=perform_managed_folder_operations, skip_if_destination_has_later_modification_time=args.skip_if_dest_has_newer_mtime, skip_unsupported=args.skip_unsupported, task_status_queue=task_status_queue)
    return task_executor.execute_tasks(operation_iterator, continue_on_error=args.continue_on_error, parallelizable=not perform_managed_folder_operations, progress_manager_args=task_status.ProgressManagerArgs(task_status.IncrementType.FILES_AND_BYTES, manifest_path=user_request_args.manifest_path), task_status_queue=task_status_queue)