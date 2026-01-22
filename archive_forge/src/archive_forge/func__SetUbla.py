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
def _SetUbla(self, blr, setting_arg):
    """Sets the Uniform bucket-level access setting for a bucket on or off."""
    self._ValidateBucketListingRefAndReturnBucketName(blr)
    bucket_url = blr.storage_url
    iam_config = IamConfigurationValue()
    iam_config.bucketPolicyOnly = uniformBucketLevelAccessValue()
    iam_config.bucketPolicyOnly.enabled = setting_arg == 'on'
    bucket_metadata = apitools_messages.Bucket(iamConfiguration=iam_config)
    setting_verb = 'Enabling' if setting_arg == 'on' else 'Disabling'
    print('%s Uniform bucket-level access for %s...' % (setting_verb, str(bucket_url).rstrip('/')))
    self.gsutil_api.PatchBucket(bucket_url.bucket_name, bucket_metadata, fields=['iamConfiguration'], provider=bucket_url.scheme)
    return 0