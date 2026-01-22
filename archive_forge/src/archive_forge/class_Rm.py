from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import name_expansion
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import rm_command_util
from googlecloudsdk.command_lib.storage import stdin_iterator
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task_iterator_factory
from googlecloudsdk.core import log
class Rm(base.Command):
    """Delete objects and buckets."""
    detailed_help = {'DESCRIPTION': '\n      Delete objects and buckets.\n      ', 'EXAMPLES': "\n\n      The following command deletes a Cloud Storage object named ``my-object''\n      from the bucket ``my-bucket'':\n\n        $ {command} gs://my-bucket/my-object\n\n      The following command deletes all objects directly within the directory\n      ``my-dir'' but no objects within subdirectories:\n\n        $ {command} gs://my-bucket/my-dir/*\n\n      The following command deletes all objects and subdirectories within the\n      directory ``my-dir'':\n\n        $ {command} gs://my-bucket/my-dir/**\n\n      Note that for buckets that contain\n      [versioned objects](https://cloud.google.com/storage/docs/object-versioning),\n      the above command only affects live versions. Use the `--recursive` flag\n      instead to delete all versions.\n\n      The following command deletes all versions of all resources in\n      ``my-bucket'' and then deletes the bucket.\n\n        $ {command} --recursive gs://my-bucket/\n\n      The following command deletes all text files in the top-level of\n      ``my-bucket'', but not text files in subdirectories:\n\n        $ {command} -recursive gs://my-bucket/*.txt\n\n      The following command deletes one wildcard expression per line passed\n      in by stdin:\n\n        $ some_program | {command} -I\n      "}

    @classmethod
    def Args(cls, parser):
        parser.add_argument('urls', nargs='*', help='The URLs of the resources to delete.')
        parser.add_argument('--recursive', '-R', '-r', action='store_true', help="Recursively delete the contents of buckets or directories that match the path expression. If the path is set to a bucket, like ``gs://bucket'', the bucket is also deleted. This option implies the `--all-versions` option. If you want to delete only live object versions, use the ``**'' wildcard instead.")
        parser.add_argument('--all-versions', '-a', action='store_true', help='Delete all [versions](https://cloud.google.com/storage/docs/object-versioning) of an object.')
        parser.add_argument('--exclude-managed-folders', action='store_true', default=False, help='Excludes managed folders from command operations. By default gcloud storage includes managed folders in recursive removals.')
        flags.add_additional_headers_flag(parser)
        flags.add_continue_on_error_flag(parser)
        flags.add_precondition_flags(parser)
        flags.add_read_paths_from_stdin_flag(parser)

    def Run(self, args):
        if args.recursive:
            bucket_setting = name_expansion.BucketSetting.YES
            object_state = cloud_api.ObjectState.LIVE_AND_NONCURRENT
            recursion_setting = name_expansion.RecursionSetting.YES
        else:
            bucket_setting = name_expansion.BucketSetting.NO
            object_state = flags.get_object_state_from_flags(args)
            recursion_setting = name_expansion.RecursionSetting.NO_WITH_WARNING
        should_perform_managed_folder_operations = args.recursive and (not args.exclude_managed_folders)
        url_found_match_tracker = collections.OrderedDict()
        name_expansion_iterator = name_expansion.NameExpansionIterator(stdin_iterator.get_urls_iterable(args.urls, args.read_paths_from_stdin), fields_scope=cloud_api.FieldsScope.SHORT, include_buckets=bucket_setting, managed_folder_setting=folder_util.ManagedFolderSetting.DO_NOT_LIST, object_state=object_state, raise_error_for_unmatched_urls=not should_perform_managed_folder_operations, recursion_requested=recursion_setting, url_found_match_tracker=url_found_match_tracker)
        user_request_args = user_request_args_factory.get_user_request_args_from_command_args(args)
        task_status_queue = task_graph_executor.multiprocessing_context.Queue()
        task_iterator_factory = delete_task_iterator_factory.DeleteTaskIteratorFactory(name_expansion_iterator, task_status_queue=task_status_queue, user_request_args=user_request_args)
        log.status.Print('Removing objects:')
        object_exit_code = task_executor.execute_tasks(task_iterator_factory.object_iterator(), parallelizable=True, task_status_queue=task_status_queue, progress_manager_args=task_status.ProgressManagerArgs(increment_type=task_status.IncrementType.INTEGER, manifest_path=None), continue_on_error=args.continue_on_error)
        if should_perform_managed_folder_operations:
            managed_folder_expansion_iterator = name_expansion.NameExpansionIterator(args.urls, managed_folder_setting=folder_util.ManagedFolderSetting.LIST_WITHOUT_OBJECTS, raise_error_for_unmatched_urls=True, raise_managed_folder_precondition_errors=False, recursion_requested=name_expansion.RecursionSetting.YES, url_found_match_tracker=url_found_match_tracker)
            try:
                managed_folder_exit_code = rm_command_util.remove_managed_folders(args, managed_folder_expansion_iterator, task_status_queue, verbose=True)
            except api_errors.GcsApiError as error:
                if error.payload.status_code != 403:
                    raise
                log.warning('Unable to delete managed folders due to missing permissions.')
                managed_folder_exit_code = 0
        else:
            managed_folder_exit_code = 0
        bucket_iterator = plurality_checkable_iterator.PluralityCheckableIterator(task_iterator_factory.bucket_iterator())
        if args.recursive and (not bucket_iterator.is_empty()):
            log.status.Print('Removing buckets:')
            bucket_exit_code = task_executor.execute_tasks(bucket_iterator, parallelizable=True, task_status_queue=task_status_queue, progress_manager_args=task_status.ProgressManagerArgs(increment_type=task_status.IncrementType.INTEGER, manifest_path=None), continue_on_error=args.continue_on_error)
        else:
            bucket_exit_code = 0
        self.exit_code = max(object_exit_code, managed_folder_exit_code, bucket_exit_code)