from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
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
def _GetRequesterPays(self):
    """Gets requesterpays configuration for one or more buckets."""
    url_args = self.args
    some_matched = False
    for url_str in url_args:
        bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['billing'])
        for blr in bucket_iter:
            some_matched = True
            if blr.root_object.billing and blr.root_object.billing.requesterPays:
                print('%s: Enabled' % blr.url_string.rstrip('/'))
            else:
                print('%s: Disabled' % blr.url_string.rstrip('/'))
    if not some_matched:
        raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))