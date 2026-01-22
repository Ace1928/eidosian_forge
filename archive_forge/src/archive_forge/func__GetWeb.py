from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
from apitools.base.py import encoding
from gslib import metrics
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
def _GetWeb(self):
    """Gets website configuration for a bucket."""
    bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(self.args[0], bucket_fields=['website'])
    if bucket_url.scheme == 's3':
        sys.stdout.write(self.gsutil_api.XmlPassThroughGetWebsite(bucket_url, provider=bucket_url.scheme))
    elif bucket_metadata.website and (bucket_metadata.website.mainPageSuffix or bucket_metadata.website.notFoundPage):
        sys.stdout.write(str(encoding.MessageToJson(bucket_metadata.website)) + '\n')
    else:
        sys.stdout.write('%s has no website configuration.\n' % bucket_url)
    return 0