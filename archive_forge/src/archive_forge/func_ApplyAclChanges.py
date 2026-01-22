from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import encoding
from gslib import metrics
from gslib import gcs_json_api
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import SetAclExceptionHandler
from gslib.command import SetAclFuncWrapper
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.help_provider import CreateHelpText
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import UrlsAreForSingleProvider
from gslib.storage_url import RaiseErrorIfUrlsAreMixOfBucketsAndObjects
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import acl_helper
from gslib.utils.constants import NO_MAX
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
@Retry(ServiceException, tries=3, timeout_secs=1)
def ApplyAclChanges(self, name_expansion_result, thread_state=None):
    """Applies the changes in self.changes to the provided URL.

    Args:
      name_expansion_result: NameExpansionResult describing the target object.
      thread_state: If present, gsutil Cloud API instance to apply the changes.
    """
    if thread_state:
        gsutil_api = thread_state
    else:
        gsutil_api = self.gsutil_api
    url = name_expansion_result.expanded_storage_url
    if url.IsBucket():
        bucket = gsutil_api.GetBucket(url.bucket_name, provider=url.scheme, fields=['acl', 'metageneration'])
        current_acl = bucket.acl
    elif url.IsObject():
        gcs_object = encoding.JsonToMessage(apitools_messages.Object, name_expansion_result.expanded_result)
        current_acl = gcs_object.acl
    if not current_acl:
        self._RaiseForAccessDenied(url)
    if self._ApplyAclChangesAndReturnChangeCount(url, current_acl) == 0:
        self.logger.info('No changes to %s', url)
        return
    try:
        if url.IsBucket():
            preconditions = Preconditions(meta_gen_match=bucket.metageneration)
            bucket_metadata = apitools_messages.Bucket(acl=current_acl)
            gsutil_api.PatchBucket(url.bucket_name, bucket_metadata, preconditions=preconditions, provider=url.scheme, fields=['id'])
        else:
            preconditions = Preconditions(gen_match=gcs_object.generation, meta_gen_match=gcs_object.metageneration)
            object_metadata = apitools_messages.Object(acl=current_acl)
            try:
                gsutil_api.PatchObjectMetadata(url.bucket_name, url.object_name, object_metadata, preconditions=preconditions, provider=url.scheme, generation=url.generation, fields=['id'])
            except PreconditionException as e:
                self._RefetchObjectMetadataAndApplyAclChanges(url, gsutil_api)
        self.logger.info('Updated ACL on %s', url)
    except BadRequestException as e:
        raise CommandException('Received bad request from server: %s' % str(e))
    except AccessDeniedException:
        self._RaiseForAccessDenied(url)
    except PreconditionException as e:
        if url.IsObject():
            raise CommandException(str(e))
        raise e