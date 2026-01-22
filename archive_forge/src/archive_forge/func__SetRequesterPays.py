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
def _SetRequesterPays(self):
    """Gets requesterpays configuration for a bucket."""
    requesterpays_arg = self.args[0].lower()
    if requesterpays_arg not in ('on', 'off'):
        raise CommandException('Argument to "%s set" must be either <on|off>' % self.command_name)
    url_args = self.args[1:]
    if not url_args:
        self.RaiseWrongNumberOfArgumentsException()
    some_matched = False
    for url_str in url_args:
        bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
        for blr in bucket_iter:
            url = blr.storage_url
            some_matched = True
            bucket_metadata = apitools_messages.Bucket(billing=apitools_messages.Bucket.BillingValue())
            if requesterpays_arg == 'on':
                self.logger.info('Enabling requester pays for %s...', url)
                bucket_metadata.billing.requesterPays = True
            else:
                self.logger.info('Disabling requester pays for %s...', url)
                bucket_metadata.billing.requesterPays = False
            self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
    if not some_matched:
        raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))