from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import errors_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import name_expansion
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import rm_command_util
from googlecloudsdk.command_lib.storage import stdin_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_iterator
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _execute_copy_tasks(args, delete_source, parallelizable, raw_destination_url, source_expansion_iterator):
    """Returns appropriate exit code after creating and executing copy tasks."""
    if raw_destination_url.is_stdio:
        task_status_queue = None
    else:
        task_status_queue = task_graph_executor.multiprocessing_context.Queue()
    user_request_args = user_request_args_factory.get_user_request_args_from_command_args(args, metadata_type=user_request_args_factory.MetadataType.OBJECT)
    with _get_shared_stream(args, raw_destination_url) as shared_stream:
        task_iterator = copy_task_iterator.CopyTaskIterator(source_expansion_iterator, args.destination, custom_md5_digest=args.content_md5, delete_source=delete_source, do_not_decompress=args.do_not_decompress, force_daisy_chain=args.daisy_chain, print_created_message=args.print_created_message, shared_stream=shared_stream, skip_unsupported=args.skip_unsupported, task_status_queue=task_status_queue, user_request_args=user_request_args)
        return task_executor.execute_tasks(task_iterator, parallelizable=parallelizable, task_status_queue=task_status_queue, progress_manager_args=task_status.ProgressManagerArgs(task_status.IncrementType.FILES_AND_BYTES, manifest_path=user_request_args.manifest_path), continue_on_error=args.continue_on_error)