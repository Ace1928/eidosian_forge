from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import encoding
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils import text_util
class LoggingCommand(Command):
    """Implementation of gsutil logging command."""
    command_spec = Command.CreateCommandSpec('logging', command_name_aliases=['disablelogging', 'enablelogging', 'getlogging'], usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='b:o:', file_url_ok=False, provider_url_ok=False, urls_start_arg=0, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument('mode', choices=['on', 'off']), CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()])
    help_spec = Command.HelpSpec(help_name='logging', help_name_aliases=['loggingconfig', 'logs', 'log', 'getlogging', 'enablelogging', 'disablelogging'], help_type='command_help', help_one_line_summary='Configure or retrieve logging on buckets', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text})
    gcloud_storage_map = GcloudStorageMap(gcloud_command={'get': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'list', '--format="gsutiljson[key=logging_config,empty=\' has no logging configuration.\',empty_prefix_key=storage_url]"', '--raw'], flag_map={}), 'set': GcloudStorageMap(gcloud_command={'on': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update'], flag_map={'-b': GcloudStorageFlag('--log-bucket'), '-o': GcloudStorageFlag('--log-object-prefix')}), 'off': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--clear-log-bucket', '--clear-log-object-prefix'], flag_map={})}, flag_map={})}, flag_map={})

    def _Get(self):
        """Gets logging configuration for a bucket."""
        bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(self.args[0], bucket_fields=['logging'])
        if bucket_url.scheme == 's3':
            text_util.print_to_fd(self.gsutil_api.XmlPassThroughGetLogging(bucket_url, provider=bucket_url.scheme), end='')
        elif bucket_metadata.logging and bucket_metadata.logging.logBucket and bucket_metadata.logging.logObjectPrefix:
            text_util.print_to_fd(str(encoding.MessageToJson(bucket_metadata.logging)))
        else:
            text_util.print_to_fd('%s has no logging configuration.' % bucket_url)
        return 0

    def _Enable(self):
        """Enables logging configuration for a bucket."""
        if not UrlsAreForSingleProvider(self.args):
            raise CommandException('"logging set on" command spanning providers not allowed.')
        target_bucket_url = None
        target_prefix = None
        for opt, opt_arg in self.sub_opts:
            if opt == '-b':
                target_bucket_url = StorageUrlFromString(opt_arg)
            if opt == '-o':
                target_prefix = opt_arg
        if not target_bucket_url:
            raise CommandException('"logging set on" requires \'-b <log_bucket>\' option')
        if not target_bucket_url.IsBucket():
            raise CommandException('-b option must specify a bucket URL.')
        some_matched = False
        for url_str in self.args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
            for blr in bucket_iter:
                url = blr.storage_url
                some_matched = True
                self.logger.info('Enabling logging on %s...', blr)
                logging = apitools_messages.Bucket.LoggingValue(logBucket=target_bucket_url.bucket_name, logObjectPrefix=target_prefix or url.bucket_name)
                bucket_metadata = apitools_messages.Bucket(logging=logging)
                self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(self.args))
        return 0

    def _Disable(self):
        """Disables logging configuration for a bucket."""
        some_matched = False
        for url_str in self.args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
            for blr in bucket_iter:
                url = blr.storage_url
                some_matched = True
                self.logger.info('Disabling logging on %s...', blr)
                logging = apitools_messages.Bucket.LoggingValue()
                bucket_metadata = apitools_messages.Bucket(logging=logging)
                self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(self.args))
        return 0

    def RunCommand(self):
        """Command entry point for the logging command."""
        action_subcommand = self.args.pop(0)
        if action_subcommand == 'get':
            func = self._Get
            metrics.LogCommandParams(subcommands=[action_subcommand])
        elif action_subcommand == 'set':
            state_subcommand = self.args.pop(0)
            if not self.args:
                self.RaiseWrongNumberOfArgumentsException()
            if state_subcommand == 'on':
                func = self._Enable
                metrics.LogCommandParams(subcommands=[action_subcommand, state_subcommand])
            elif state_subcommand == 'off':
                func = self._Disable
                metrics.LogCommandParams(subcommands=[action_subcommand, state_subcommand])
            else:
                raise CommandException('Invalid subcommand "%s" for the "%s %s" command.\nSee "gsutil help logging".' % (state_subcommand, self.command_name, action_subcommand))
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.\nSee "gsutil help logging".' % (action_subcommand, self.command_name))
        self.ParseSubOpts(check_args=True)
        metrics.LogCommandParams(sub_opts=self.sub_opts)
        func()
        return 0