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