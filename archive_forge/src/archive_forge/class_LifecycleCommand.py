from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import LifecycleTranslation
class LifecycleCommand(Command):
    """Implementation of gsutil lifecycle command."""
    command_spec = Command.CreateCommandSpec('lifecycle', command_name_aliases=['lifecycleconfig'], usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='', file_url_ok=True, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.JSON, ApiSelector.XML], gs_default_api=ApiSelector.JSON, argparse_arguments={'set': [CommandArgument.MakeNFileURLsArgument(1), CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'get': [CommandArgument.MakeNCloudBucketURLsArgument(1)]})
    help_spec = Command.HelpSpec(help_name='lifecycle', help_name_aliases=['getlifecycle', 'setlifecycle'], help_type='command_help', help_one_line_summary='Get or set lifecycle configuration for a bucket', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text})

    def get_gcloud_storage_args(self):
        if self.args[0] == 'set':
            gcloud_storage_map = GcloudStorageMap(gcloud_command={'set': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--lifecycle-file={}'.format(self.args[1])] + self.args[2:], flag_map={})}, flag_map={})
            self.args = self.args[:1]
        else:
            gcloud_storage_map = GcloudStorageMap(gcloud_command={'get': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'describe', '--format="gsutiljson[key=lifecycle_config,empty=\' has no lifecycle configuration.\',empty_prefix_key=storage_url]"', '--raw'], flag_map={})}, flag_map={})
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def _SetLifecycleConfig(self):
        """Sets lifecycle configuration for a Google Cloud Storage bucket."""
        lifecycle_arg = self.args[0]
        url_args = self.args[1:]
        if not UrlsAreForSingleProvider(url_args):
            raise CommandException('"%s" command spanning providers not allowed.' % self.command_name)
        lifecycle_file = open(lifecycle_arg, 'r')
        lifecycle_txt = lifecycle_file.read()
        lifecycle_file.close()
        some_matched = False
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['lifecycle'])
            for blr in bucket_iter:
                url = blr.storage_url
                some_matched = True
                self.logger.info('Setting lifecycle configuration on %s...', blr)
                if url.scheme == 's3':
                    self.gsutil_api.XmlPassThroughSetLifecycle(lifecycle_txt, url, provider=url.scheme)
                else:
                    lifecycle = LifecycleTranslation.JsonLifecycleToMessage(lifecycle_txt)
                    bucket_metadata = apitools_messages.Bucket(lifecycle=lifecycle)
                    self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
        return 0

    def _GetLifecycleConfig(self):
        """Gets lifecycle configuration for a Google Cloud Storage bucket."""
        bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(self.args[0], bucket_fields=['lifecycle'])
        if bucket_url.scheme == 's3':
            sys.stdout.write(self.gsutil_api.XmlPassThroughGetLifecycle(bucket_url, provider=bucket_url.scheme))
        elif bucket_metadata.lifecycle and bucket_metadata.lifecycle.rule:
            sys.stdout.write(LifecycleTranslation.JsonLifecycleFromMessage(bucket_metadata.lifecycle))
        else:
            sys.stdout.write('%s has no lifecycle configuration.\n' % bucket_url)
        return 0

    def RunCommand(self):
        """Command entry point for the lifecycle command."""
        subcommand = self.args.pop(0)
        if subcommand == 'get':
            metrics.LogCommandParams(subcommands=[subcommand])
            return self._GetLifecycleConfig()
        elif subcommand == 'set':
            metrics.LogCommandParams(subcommands=[subcommand])
            return self._SetLifecycleConfig()
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.' % (subcommand, self.command_name))