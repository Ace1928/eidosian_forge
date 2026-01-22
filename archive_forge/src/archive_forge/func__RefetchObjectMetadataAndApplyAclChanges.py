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
@Retry(PreconditionException, tries=3, timeout_secs=1)
def _RefetchObjectMetadataAndApplyAclChanges(self, url, gsutil_api):
    """Reattempts object ACL changes after a PreconditionException."""
    gcs_object = gsutil_api.GetObjectMetadata(url.bucket_name, url.object_name, provider=url.scheme, fields=['acl', 'generation', 'metageneration'])
    current_acl = gcs_object.acl
    if self._ApplyAclChangesAndReturnChangeCount(url, current_acl) == 0:
        self.logger.info('No changes to %s', url)
        return
    object_metadata = apitools_messages.Object(acl=current_acl)
    preconditions = Preconditions(gen_match=gcs_object.generation, meta_gen_match=gcs_object.metageneration)
    gsutil_api.PatchObjectMetadata(url.bucket_name, url.object_name, object_metadata, preconditions=preconditions, provider=url.scheme, generation=gcs_object.generation, fields=['id'])