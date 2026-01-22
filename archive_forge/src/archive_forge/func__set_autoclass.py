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
def _set_autoclass(self, blr, setting_arg):
    """Turns autoclass on or off for a bucket."""
    bucket_url = blr.storage_url
    autoclass_config = apitools_messages.Bucket.AutoclassValue()
    autoclass_config.enabled = setting_arg == 'on'
    bucket_metadata = apitools_messages.Bucket(autoclass=autoclass_config)
    print('Setting Autoclass %s for %s' % (setting_arg, str(bucket_url).rstrip('/')))
    self.gsutil_api.PatchBucket(bucket_url.bucket_name, bucket_metadata, fields=['autoclass'], provider=bucket_url.scheme)
    return 0