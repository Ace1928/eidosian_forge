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
def _SetWeb(self):
    """Sets website configuration for a bucket."""
    main_page_suffix = None
    error_page = None
    if self.sub_opts:
        for o, a in self.sub_opts:
            if o == '-m':
                main_page_suffix = a
            elif o == '-e':
                error_page = a
    url_args = self.args
    website = apitools_messages.Bucket.WebsiteValue(mainPageSuffix=main_page_suffix, notFoundPage=error_page)
    some_matched = False
    for url_str in url_args:
        bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
        for blr in bucket_iter:
            url = blr.storage_url
            some_matched = True
            self.logger.info('Setting website configuration on %s...', blr)
            bucket_metadata = apitools_messages.Bucket(website=website)
            self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
    if not some_matched:
        raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
    return 0