from __future__ import absolute_import
from __future__ import print_function
import getopt
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
from gslib.utils.text_util import InsistOnOrOff
from gslib.utils import shim_util
def _GetUbla(self, blr):
    """Gets the Uniform bucket-level access setting for a bucket."""
    self._ValidateBucketListingRefAndReturnBucketName(blr)
    bucket_url = blr.storage_url
    bucket_metadata = self.gsutil_api.GetBucket(bucket_url.bucket_name, fields=['iamConfiguration'], provider=bucket_url.scheme)
    iam_config = bucket_metadata.iamConfiguration
    uniform_bucket_level_access = iam_config.bucketPolicyOnly
    fields = {'bucket': str(bucket_url).rstrip('/'), 'enabled': uniform_bucket_level_access.enabled}
    locked_time_line = ''
    if uniform_bucket_level_access.lockedTime:
        fields['locked_time'] = uniform_bucket_level_access.lockedTime
        locked_time_line = '  LockedTime: {locked_time}\n'
    if uniform_bucket_level_access:
        print(('Uniform bucket-level access setting for {bucket}:\n  Enabled: {enabled}\n' + locked_time_line).format(**fields))