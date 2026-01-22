from __future__ import absolute_import
from __future__ import print_function
import getopt
import textwrap
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.commands import ubla
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.text_util import InsistOnOrOff
class BucketPolicyOnlyCommand(Command):
    """Implements the gsutil bucketpolicyonly command."""
    command_spec = Command.CreateCommandSpec('bucketpolicyonly', usage_synopsis=_SYNOPSIS, min_args=2, max_args=NO_MAX, supported_sub_args='', file_url_ok=False, provider_url_ok=False, urls_start_arg=2, gs_api_support=[ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments={'get': [CommandArgument.MakeNCloudURLsArgument(1)], 'set': [CommandArgument('mode', choices=['on', 'off']), CommandArgument.MakeZeroOrMoreCloudBucketURLsArgument()]})
    help_spec = Command.HelpSpec(help_name='bucketpolicyonly', help_name_aliases=[], help_type='command_help', help_one_line_summary='Configure uniform bucket-level access', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={'get': _get_help_text, 'set': _set_help_text})
    gcloud_storage_map = ubla.UblaCommand.gcloud_storage_map
    format_flag = gcloud_storage_map.gcloud_command['get'].gcloud_command[3]
    gcloud_storage_map.gcloud_command['get'].gcloud_command[3] = format_flag.replace('Uniform bucket-level access', 'Bucket Policy Only')

    def _ValidateBucketListingRefAndReturnBucketName(self, blr):
        if blr.storage_url.scheme != 'gs':
            raise CommandException('The %s command can only be used with gs:// bucket URLs.' % self.command_name)

    def _GetBucketPolicyOnly(self, blr):
        """Gets the Bucket Policy Only setting for a bucket."""
        self._ValidateBucketListingRefAndReturnBucketName(blr)
        bucket_url = blr.storage_url
        bucket_metadata = self.gsutil_api.GetBucket(bucket_url.bucket_name, fields=['iamConfiguration'], provider=bucket_url.scheme)
        iam_config = bucket_metadata.iamConfiguration
        bucket_policy_only = iam_config.bucketPolicyOnly
        fields = {'bucket': str(bucket_url).rstrip('/'), 'enabled': bucket_policy_only.enabled}
        locked_time_line = ''
        if bucket_policy_only.lockedTime:
            fields['locked_time'] = bucket_policy_only.lockedTime
            locked_time_line = '  LockedTime: {locked_time}\n'
        if bucket_policy_only:
            print(('Bucket Policy Only setting for {bucket}:\n  Enabled: {enabled}\n' + locked_time_line).format(**fields))

    def _SetBucketPolicyOnly(self, blr, setting_arg):
        """Sets the Bucket Policy Only setting for a bucket on or off."""
        self._ValidateBucketListingRefAndReturnBucketName(blr)
        bucket_url = blr.storage_url
        iam_config = IamConfigurationValue()
        iam_config.bucketPolicyOnly = BucketPolicyOnlyValue()
        iam_config.bucketPolicyOnly.enabled = setting_arg == 'on'
        bucket_metadata = apitools_messages.Bucket(iamConfiguration=iam_config)
        setting_verb = 'Enabling' if setting_arg == 'on' else 'Disabling'
        print('%s Bucket Policy Only for %s...' % (setting_verb, str(bucket_url).rstrip('/')))
        self.gsutil_api.PatchBucket(bucket_url.bucket_name, bucket_metadata, fields=['iamConfiguration'], provider=bucket_url.scheme)
        return 0

    def _BucketPolicyOnly(self):
        """Handles bucketpolicyonly command on a Cloud Storage bucket."""
        subcommand = self.args.pop(0)
        if subcommand not in ('get', 'set'):
            raise CommandException('bucketpolicyonly only supports get|set')
        subcommand_func = None
        subcommand_args = []
        setting_arg = None
        if subcommand == 'get':
            subcommand_func = self._GetBucketPolicyOnly
        elif subcommand == 'set':
            subcommand_func = self._SetBucketPolicyOnly
            setting_arg = self.args.pop(0)
            InsistOnOrOff(setting_arg, 'Only on and off values allowed for set option')
            subcommand_args.append(setting_arg)
        some_matched = False
        url_args = self.args
        if not url_args:
            self.RaiseWrongNumberOfArgumentsException()
        for url_str in url_args:
            bucket_iter = self.GetBucketUrlIterFromArg(url_str)
            for bucket_listing_ref in bucket_iter:
                some_matched = True
                subcommand_func(bucket_listing_ref, *subcommand_args)
        if not some_matched:
            raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
        return 0

    def RunCommand(self):
        """Command entry point for the bucketpolicyonly command."""
        if self.gsutil_api.GetApiSelector(provider='gs') != ApiSelector.JSON:
            raise CommandException('\n'.join(textwrap.wrap('The "%s" command can only be used with the Cloud Storage JSON API.' % self.command_name)))
        action_subcommand = self.args[0]
        self.ParseSubOpts(check_args=True)
        if action_subcommand == 'get' or action_subcommand == 'set':
            metrics.LogCommandParams(sub_opts=self.sub_opts)
            metrics.LogCommandParams(subcommands=[action_subcommand])
            self._BucketPolicyOnly()
        else:
            raise CommandException('Invalid subcommand "%s", use get|set instead.' % action_subcommand)