from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import codecs
import json
import os
import six
from gslib import metrics
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import UTF8
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import LabelTranslation
def _GetAndPrintLabel(self, bucket_arg):
    """Gets and prints the labels for a cloud bucket."""
    bucket_url, bucket_metadata = self.GetSingleBucketUrlFromArg(bucket_arg, bucket_fields=['labels'])
    if bucket_url.scheme == 's3':
        print(self.gsutil_api.XmlPassThroughGetTagging(bucket_url, provider=bucket_url.scheme))
    elif bucket_metadata.labels:
        print(LabelTranslation.JsonFromMessage(bucket_metadata.labels, pretty_print=True))
    else:
        print('%s has no label configuration.' % bucket_url)