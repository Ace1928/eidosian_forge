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
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import NO_MAX
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils import text_util
def _Get(self):
    """Gets logging configuration for a bucket."""
    bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(self.args[0], bucket_fields=['logging'])
    if bucket_url.scheme == 's3':
        text_util.print_to_fd(self.gsutil_api.XmlPassThroughGetLogging(bucket_url, provider=bucket_url.scheme), end='')
    elif bucket_metadata.logging and bucket_metadata.logging.logBucket and bucket_metadata.logging.logObjectPrefix:
        text_util.print_to_fd(str(encoding.MessageToJson(bucket_metadata.logging)))
    else:
        text_util.print_to_fd('%s has no logging configuration.' % bucket_url)
    return 0