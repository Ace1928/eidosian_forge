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
def _SetRpo(self, blr, rpo_value):
    """Sets the rpo setting for a bucket."""
    bucket_url = blr.storage_url
    formatted_rpo_value = rpo_value
    if formatted_rpo_value not in VALID_RPO_VALUES:
        raise CommandException('Invalid value for rpo set. Should be one of {}'.format(VALID_RPO_VALUES_STRING))
    bucket_metadata = apitools_messages.Bucket(rpo=formatted_rpo_value)
    self.logger.info('Setting rpo %s for %s' % (formatted_rpo_value, str(bucket_url).rstrip('/')))
    self.gsutil_api.PatchBucket(bucket_url.bucket_name, bucket_metadata, fields=['rpo'], provider=bucket_url.scheme)
    return 0