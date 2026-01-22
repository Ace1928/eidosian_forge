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
@Retry(PreconditionException, tries=3, timeout_secs=1)
def _SetLabelForBucket(blr):
    url = blr.storage_url
    self.logger.info('Setting label configuration on %s...', blr)
    if url.scheme == 's3':
        self.gsutil_api.XmlPassThroughSetTagging(label_text, url, provider=url.scheme)
    else:
        labels_message = None
        metageneration = None
        new_label_json = json.loads(label_text)
        if self.gsutil_api.GetApiSelector(url.scheme) == ApiSelector.JSON:
            _, bucket_metadata = self.GetSingleBucketUrlFromArg(url.url_string, bucket_fields=['labels', 'metageneration'])
            metageneration = bucket_metadata.metageneration
            label_json = {}
            if bucket_metadata.labels:
                label_json = json.loads(LabelTranslation.JsonFromMessage(bucket_metadata.labels))
            merged_labels = dict(((key, None) for key, _ in six.iteritems(label_json)))
            merged_labels.update(new_label_json)
            labels_message = LabelTranslation.DictToMessage(merged_labels)
        else:
            labels_message = LabelTranslation.DictToMessage(new_label_json)
        preconditions = Preconditions(meta_gen_match=metageneration)
        bucket_metadata = apitools_messages.Bucket(labels=labels_message)
        self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, preconditions=preconditions, provider=url.scheme, fields=['id'])