from __future__ import absolute_import
import time
from apitools.base.py import encoding
from gslib import metrics
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import Preconditions
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.help_provider import CreateHelpText
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import MetadataMessage
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import NO_MAX
from gslib.utils.parallelism_framework_util import PutToQueueWithTimeout
from gslib.utils.retention_util import ConfirmLockRequest
from gslib.utils.retention_util import ReleaseEventHoldFuncWrapper
from gslib.utils.retention_util import ReleaseTempHoldFuncWrapper
from gslib.utils.retention_util import RetentionInSeconds
from gslib.utils.retention_util import RetentionPolicyToString
from gslib.utils.retention_util import SetEventHoldFuncWrapper
from gslib.utils.retention_util import SetTempHoldFuncWrapper
from gslib.utils.retention_util import UpdateObjectMetadataExceptionHandler
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.translation_helper import PreconditionsFromHeaders
def _LockRetention(self):
    """Lock Retention Policy on one or more buckets."""
    url_args = self.args
    some_matched = False
    for url_str in url_args:
        bucket_iter = self.GetBucketUrlIterFromArg(url_str, bucket_fields=['id'])
        for blr in bucket_iter:
            url = blr.storage_url
            some_matched = True
            bucket_metadata = self.gsutil_api.GetBucket(url.bucket_name, provider=url.scheme, fields=['id', 'metageneration', 'retentionPolicy'])
            if not (bucket_metadata.retentionPolicy and bucket_metadata.retentionPolicy.retentionPeriod):
                raise CommandException('Bucket "{}" does not have an Unlocked Retention Policy.'.format(url.bucket_name))
            elif bucket_metadata.retentionPolicy.isLocked is True:
                self.logger.error('Retention Policy on "%s" is already locked.', blr)
            elif ConfirmLockRequest(url.bucket_name, bucket_metadata.retentionPolicy):
                self.logger.info('Locking Retention Policy on %s...', blr)
                self.gsutil_api.LockRetentionPolicy(url.bucket_name, bucket_metadata.metageneration, provider=url.scheme)
            else:
                self.logger.error('  Abort Locking Retention Policy on {}'.format(blr))
    if not some_matched:
        raise CommandException(NO_URLS_MATCHED_TARGET % list(url_args))
    return 0