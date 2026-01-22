from __future__ import absolute_import
from __future__ import print_function
import textwrap
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils import shim_util
class PapCommand(Command):
    """Implements the gsutil pap command."""
    command_spec = Command.CreateCommandSpec('pap', command_name_aliases=['publicaccessprevention'], usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='', file_url_ok=False, provider_url_ok=False, urls_start_arg=2, gs_api_support=[ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'get': [CommandArgument.MakeNCloudURLsArgument(1)], 'set': [CommandArgument('mode', choices=['enforced', 'inherited']), CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()]})
    help_spec = Command.HelpSpec(help_name='pap', help_name_aliases=['publicaccessprevention'], help_type='command_help', help_one_line_summary='Configure public access prevention', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text})
    gcloud_storage_map = GcloudStorageMap(gcloud_command={'get': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'list', _GCLOUD_LIST_FORMAT, '--raw'], flag_map={}, supports_output_translation=True), 'set': GcloudStorageMap(gcloud_command={'enforced': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--public-access-prevention'], flag_map={}), 'inherited': GcloudStorageMap(gcloud_command=['storage', 'buckets', 'update', '--no-public-access-prevention'], flag_map={})}, flag_map={})}, flag_map={})

    def _ValidateBucketListingRefAndReturnBucketName(self, blr):
        if blr.storage_url.scheme != 'gs':
            raise CommandException('The %s command can only be used with gs:// bucket URLs.' % self.command_name)

    def _GetPublicAccessPrevention(self, blr):
        """Gets the public access prevention setting for a bucket."""
        bucket_url = blr.storage_url
        bucket_metadata = self.gsutil_api.GetBucket(bucket_url.bucket_name, fields=['iamConfiguration'], provider=bucket_url.scheme)
        iam_config = bucket_metadata.iamConfiguration
        public_access_prevention = iam_config.publicAccessPrevention or 'inherited'
        bucket = str(bucket_url).rstrip('/')
        print('%s: %s' % (bucket, public_access_prevention))

    def _SetPublicAccessPrevention(self, blr, setting_arg):
        """Sets the Public Access Prevention setting for a bucket enforced or inherited."""
        bucket_url = blr.storage_url
        iam_config = IamConfigurationValue()
        iam_config.publicAccessPrevention = setting_arg
        bucket_metadata = apitools_messages.Bucket(iamConfiguration=iam_config)
        print('Setting Public Access Prevention %s for %s' % (setting_arg, str(bucket_url).rstrip('/')))
        self.gsutil_api.PatchBucket(bucket_url.bucket_name, bucket_metadata, fields=['iamConfiguration'], provider=bucket_url.scheme)
        return 0

    def _Pap(self):
        """Handles pap command on Cloud Storage buckets."""
        subcommand = self.args.pop(0)
        if subcommand not in ('get', 'set'):
            raise CommandException('pap only supports get|set')
        subcommand_func = None
        subcommand_args = []
        setting_arg = None
        if subcommand == 'get':
            subcommand_func = self._GetPublicAccessPrevention
        elif subcommand == 'set':
            subcommand_func = self._SetPublicAccessPrevention
            setting_arg = self.args.pop(0)
            subcommand_args.append(setting_arg)
        if self.gsutil_api.GetApiSelector('gs') != ApiSelector.JSON:
            raise CommandException('\n'.join(textwrap.wrap('The "%s" command can only be with the Cloud Storage JSON API.' % self.command_name)))
        some_matched = False
        url_args = self.args
        if not url_args:
            self.RaiseWrongNumberOfArgumentsException()
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str)
            for bucket_listing_ref in bucket_iter:
                if self.gsutil_api.GetApiSelector(bucket_listing_ref.storage_url.scheme) != ApiSelector.JSON:
                    raise CommandException('\n'.join(textwrap.wrap('The "%s" command can only be used for GCS Buckets.' % self.command_name)))
                some_matched = True
                subcommand_func(bucket_listing_ref, *subcommand_args)
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
        return 0

    def RunCommand(self):
        """Command entry point for the pap command."""
        action_subcommand = self.args[0]
        self.ParseSubOpts(check_args=True)
        if action_subcommand == 'get' or action_subcommand == 'set':
            metrics.LogCommandParams(sub_opts=self.sub_opts)
            metrics.LogCommandParams(subcommands=[action_subcommand])
            self._Pap()
        else:
            raise CommandException('Invalid subcommand "%s", use get|set instead.' % action_subcommand)