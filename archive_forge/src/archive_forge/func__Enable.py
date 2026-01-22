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
def _Enable(self):
    """Enables logging configuration for a bucket."""
    if not UrlsAreForSingleProvider(self.args):
        raise CommandException('"logging set on" command spanning providers not allowed.')
    target_bucket_url = None
    target_prefix = None
    for opt, opt_arg in self.sub_opts:
        if opt == '-b':
            target_bucket_url = StorageUrlFromString(opt_arg)
        if opt == '-o':
            target_prefix = opt_arg
    if not target_bucket_url:
        raise CommandException('"logging set on" requires \'-b <log_bucket>\' option')
    if not target_bucket_url.IsBucket():
        raise CommandException('-b option must specify a bucket URL.')
    some_matched = False
    for url_str in self.args:
        bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
        for blr in bucket_iter:
            url = blr.storage_url
            some_matched = True
            self.logger.info('Enabling logging on %s...', blr)
            logging = apitools_messages.Bucket.LoggingValue(logBucket=target_bucket_url.bucket_name, logObjectPrefix=target_prefix or url.bucket_name)
            bucket_metadata = apitools_messages.Bucket(logging=logging)
            self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
    if not some_matched:
        raise CommandException(NO_URLS_MATCHED_TARGET % list(self.args))
    return 0