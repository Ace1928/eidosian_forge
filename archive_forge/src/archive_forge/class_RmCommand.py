from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import time
from gslib.cloud_api import BucketNotFoundException
from gslib.cloud_api import NotEmptyException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import DecrementFailureCount
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.storage_url import StorageUrlFromString
from gslib.thread_message import MetadataMessage
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.system_util import StdinIterator
from gslib.utils.translation_helper import PreconditionsFromHeaders
class RmCommand(Command):
    """Implementation of gsutil rm command."""
    command_spec = Command.CreateCommandSpec('rm', command_name_aliases=['del', 'delete', 'remove'], usage_synopsis=_SYNOPSIS, min_args=0, max_args=constants.NO_MAX, supported_sub_args='afIrR', file_url_ok=False, provider_url_ok=False, urls_start_arg=0, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeZeroOrMoreCloudURLsArgument()])
    help_spec = Command.HelpSpec(help_name='rm', help_name_aliases=['del', 'delete', 'remove'], help_type='command_help', help_one_line_summary='Remove objects', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})
    gcloud_storage_map = GcloudStorageMap(gcloud_command=['storage', 'rm'], flag_map={'-r': GcloudStorageFlag('-r'), '-R': GcloudStorageFlag('-r'), '-a': GcloudStorageFlag('-a'), '-I': GcloudStorageFlag('-I'), '-f': GcloudStorageFlag('--continue-on-error')})

    def RunCommand(self):
        """Command entry point for the rm command."""
        self.continue_on_error = self.parallel_operations
        self.read_args_from_stdin = False
        self.all_versions = False
        if self.sub_opts:
            for o, unused_a in self.sub_opts:
                if o == '-a':
                    self.all_versions = True
                elif o == '-f':
                    self.continue_on_error = True
                elif o == '-I':
                    self.read_args_from_stdin = True
                elif o == '-r' or o == '-R':
                    self.recursion_requested = True
                    self.all_versions = True
        if self.read_args_from_stdin:
            if self.args:
                raise CommandException('No arguments allowed with the -I flag.')
            url_strs = StdinIterator()
        else:
            if not self.args:
                raise CommandException('The rm command (without -I) expects at least one URL.')
            url_strs = self.args
        self.op_failure_count = 0
        self.bucket_not_found_count = 0
        bucket_urls_to_delete = []
        self.bucket_strings_to_delete = []
        if self.recursion_requested:
            bucket_fields = ['id']
            for url_str in url_strs:
                url = StorageUrlFromString(url_str)
                if url.IsBucket() or url.IsProvider():
                    for blr in self.WildcardIterator(url_str).IterBuckets(bucket_fields=bucket_fields):
                        bucket_urls_to_delete.append(blr.storage_url)
                        self.bucket_strings_to_delete.append(url_str)
        self.preconditions = PreconditionsFromHeaders(self.headers or {})
        try:
            name_expansion_iterator = NameExpansionIterator(self.command_name, self.debug, self.logger, self.gsutil_api, url_strs, self.recursion_requested, project_id=self.project_id, all_versions=self.all_versions, continue_on_error=self.continue_on_error or self.parallel_operations)
            seek_ahead_iterator = None
            if not self.read_args_from_stdin:
                seek_ahead_iterator = SeekAheadNameExpansionIterator(self.command_name, self.debug, self.GetSeekAheadGsutilApi(), url_strs, self.recursion_requested, all_versions=self.all_versions, project_id=self.project_id)
            self.Apply(_RemoveFuncWrapper, name_expansion_iterator, _RemoveExceptionHandler, fail_on_error=not self.continue_on_error, shared_attrs=['op_failure_count', 'bucket_not_found_count'], seek_ahead_iterator=seek_ahead_iterator)
        except CommandException as e:
            if _ExceptionMatchesBucketToDelete(self.bucket_strings_to_delete, e):
                DecrementFailureCount()
            else:
                raise
        except ServiceException as e:
            if not self.continue_on_error:
                raise
        if self.bucket_not_found_count:
            raise CommandException('Encountered non-existent bucket during listing')
        if self.op_failure_count and (not self.continue_on_error):
            raise CommandException('Some files could not be removed.')
        if self.recursion_requested:
            folder_object_wildcards = []
            for url_str in url_strs:
                url = StorageUrlFromString(url_str)
                if url.IsObject():
                    folder_object_wildcards.append(url_str.rstrip('*') + '*_$folder$')
            if folder_object_wildcards:
                self.continue_on_error = True
                try:
                    name_expansion_iterator = NameExpansionIterator(self.command_name, self.debug, self.logger, self.gsutil_api, folder_object_wildcards, self.recursion_requested, project_id=self.project_id, all_versions=self.all_versions)
                    self.Apply(_RemoveFuncWrapper, name_expansion_iterator, _RemoveFoldersExceptionHandler, fail_on_error=False)
                except CommandException as e:
                    if not e.reason.startswith(NO_URLS_MATCHED_PREFIX):
                        raise
        for url in bucket_urls_to_delete:
            self.logger.info('Removing %s...', url)

            @Retry(NotEmptyException, tries=3, timeout_secs=1)
            def BucketDeleteWithRetry():
                self.gsutil_api.DeleteBucket(url.bucket_name, provider=url.scheme)
            BucketDeleteWithRetry()
        if self.op_failure_count:
            plural_str = 's' if self.op_failure_count else ''
            raise CommandException('%d file%s/object%s could not be removed.' % (self.op_failure_count, plural_str, plural_str))
        return 0

    def RemoveFunc(self, name_expansion_result, thread_state=None):
        gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
        exp_src_url = name_expansion_result.expanded_storage_url
        self.logger.info('Removing %s...', exp_src_url)
        try:
            gsutil_api.DeleteObject(exp_src_url.bucket_name, exp_src_url.object_name, preconditions=self.preconditions, generation=exp_src_url.generation, provider=exp_src_url.scheme)
        except NotFoundException as e:
            self.logger.info('Cannot find %s', exp_src_url)
            DecrementFailureCount()
        _PutToQueueWithTimeout(gsutil_api.status_queue, MetadataMessage(message_time=time.time()))