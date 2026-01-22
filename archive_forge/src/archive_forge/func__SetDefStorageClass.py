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
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils import shim_util
def _SetDefStorageClass(self):
    """Sets the default storage class for a bucket."""
    normalized_storage_class = NormalizeStorageClass(self.args[0])
    url_args = self.args[1:]
    if not url_args:
        self.RaiseWrongNumberOfArgumentsException()
    some_matched = False
    for url_str in url_args:
        self._CheckIsGsUrl(url_str)
        bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
        for blr in bucket_iter:
            some_matched = True
            bucket_metadata = apitools_messages.Bucket()
            self.logger.info('Setting default storage class to "%s" for bucket %s' % (normalized_storage_class, blr.url_string.rstrip('/')))
            bucket_metadata.storageClass = normalized_storage_class
            self.gsutil_api.PatchBucket(blr.storage_url.bucket_name, bucket_metadata, provider=blr.storage_url.scheme, fields=['id'])
    if not some_matched:
        raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))