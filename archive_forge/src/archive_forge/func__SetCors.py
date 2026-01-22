from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
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
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import CorsTranslation
from gslib.utils.translation_helper import REMOVE_CORS_CONFIG
def _SetCors(self):
    """Sets CORS configuration on a Google Cloud Storage bucket."""
    cors_arg = self.args[0]
    url_args = self.args[1:]
    if not UrlsAreForSingleProvider(url_args):
        raise CommandException('"%s" command spanning providers not allowed.' % self.command_name)
    cors_file = open(cors_arg, 'r')
    cors_txt = cors_file.read()
    cors_file.close()
    self.api = self.gsutil_api.GetApiSelector(StorageUrlFromString(url_args[0]).scheme)
    some_matched = False
    for url_str in url_args:
        bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
        for blr in bucket_iter:
            url = blr.storage_url
            some_matched = True
            self.logger.info('Setting CORS on %s...', blr)
            if url.scheme == 's3':
                self.gsutil_api.XmlPassThroughSetCors(cors_txt, url, provider=url.scheme)
            else:
                cors = CorsTranslation.JsonCorsToMessageEntries(cors_txt)
                if not cors:
                    cors = REMOVE_CORS_CONFIG
                bucket_metadata = apitools_messages.Bucket(cors=cors)
                self.gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, provider=url.scheme, fields=['id'])
    if not some_matched:
        raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
    return 0