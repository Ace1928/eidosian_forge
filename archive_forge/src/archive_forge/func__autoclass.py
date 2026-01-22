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
from gslib.utils import text_util
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils import shim_util
def _autoclass(self):
    """Handles autoclass command on Cloud Storage buckets."""
    subcommand = self.args.pop(0)
    if subcommand not in ('get', 'set'):
        raise CommandException('autoclass only supports get|set')
    subcommand_func = None
    subcommand_args = []
    setting_arg = None
    if subcommand == 'get':
        subcommand_func = self._get_autoclass
    elif subcommand == 'set':
        subcommand_func = self._set_autoclass
        setting_arg = self.args.pop(0)
        text_util.InsistOnOrOff(setting_arg, 'Only on and off values allowed for set option')
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