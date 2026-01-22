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
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import CorsTranslation
from gslib.utils.translation_helper import REMOVE_CORS_CONFIG
class CorsCommand(Command):
    """Implementation of gsutil cors command."""
    command_spec = Command.CreateCommandSpec('cors', command_name_aliases=['getcors', 'setcors'], usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='', file_url_ok=False, provider_url_ok=False, urls_start_arg=1, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'set': [CommandArgument.MakeNFileURLsArgument(1), CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()], 'get': [CommandArgument.MakeNCloudBucketURLsArgument(1)]})
    help_spec = Command.HelpSpec(help_name='cors', help_name_aliases=['getcors', 'setcors', 'cross-origin'], help_type='command_help', help_one_line_summary='Get or set a CORS configuration for one or more buckets', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text})
    gcloud_storage_map = GcloudStorageMap(gcloud_command={'get': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'describe', '--format="gsutiljson[key=cors_config,empty=\' has no CORS configuration.\',empty_prefix_key=storage_url]"', '--raw'], flag_map={}), 'set': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--cors-file'], flag_map={})}, flag_map={})

    def _CalculateUrlsStartArg(self):
        if not self.args:
            self.RaiseWrongNumberOfArgumentsException()
        if self.args[0].lower() == 'set':
            return 2
        else:
            return 1

    def _SetCors(self):
        """Sets CORS configuration on a Google Cloud Storage bucket."""
        cors_arg = self.args[0]
        url_args = self.args[1:]
        if not UrlsAreForSingleProvider(url_args):
            raise CommandException('"%s" command spanning providers not allowed.' % self.command_name)
        cors_file = open(cors_arg, 'r')
        cors_txt = cors_file.read()
        cors_file.close()
        self.api = self.gsutil_api.GetApiSelector(StorageUrlFromString(url_args[0]).scheme)
        some_matched = False
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
            for blr in bucket_iter:
                url = blr.storage_url
                some_matched = True
                self.logger.info('Setting CORS on %s...', blr)
                if url.scheme == 's3':
                    self.gsutil_api.XmlPassThroughSetCors(cors_txt, url, provider=url.scheme)
                else:
                    cors = CorsTranslation.JsonCorsToMessageEntries(cors_txt)
                    if not cors:
                        cors = REMOVE_CORS_CONFIG
                    bucket_metadata = apitools_messages.Bucket(cors=cors)
                    self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
        return 0

    def _GetCors(self):
        """Gets CORS configuration for a Google Cloud Storage bucket."""
        bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(self.args[0], bucket_fields=['cors'])
        if bucket_url.scheme == 's3':
            sys.stdout.write(self.gsutil_api.XmlPassThroughGetCors(bucket_url, provider=bucket_url.scheme))
        elif bucket_metadata.cors:
            sys.stdout.write(CorsTranslation.MessageEntriesToJson(bucket_metadata.cors))
        else:
            sys.stdout.write('%s has no CORS configuration.\n' % bucket_url)
        return 0

    def RunCommand(self):
        """Command entry point for the cors command."""
        action_subcommand = self.args.pop(0)
        if action_subcommand == 'get':
            func = self._GetCors
        elif action_subcommand == 'set':
            func = self._SetCors
        else:
            raise CommandException('Invalid subcommand "%s" for the %s command.\nSee "gsutil help cors".' % (action_subcommand, self.command_name))
        metrics.LogCommandParams(subcommands=[action_subcommand])
        return func()