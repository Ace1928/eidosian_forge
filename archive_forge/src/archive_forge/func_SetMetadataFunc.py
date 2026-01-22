from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import time
from apitools.base.py import encoding
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import MetadataMessage
from gslib.utils import constants
from gslib.utils import parallelism_framework_util
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.metadata_util import IsCustomMetadataHeader
from gslib.utils.retry_util import Retry
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.text_util import InsistAsciiHeader
from gslib.utils.text_util import InsistAsciiHeaderValue
from gslib.utils.translation_helper import CopyObjectMetadata
from gslib.utils.translation_helper import ObjectMetadataFromHeaders
from gslib.utils.translation_helper import PreconditionsFromHeaders
@Retry(PreconditionException, tries=3, timeout_secs=1)
def SetMetadataFunc(self, name_expansion_result, thread_state=None):
    """Sets metadata on an object.

    Args:
      name_expansion_result: NameExpansionResult describing target object.
      thread_state: gsutil Cloud API instance to use for the operation.
    """
    gsutil_api = GetCloudApiInstance(self, thread_state=thread_state)
    exp_src_url = name_expansion_result.expanded_storage_url
    self.logger.info('Setting metadata on %s...', exp_src_url)
    cloud_obj_metadata = encoding.JsonToMessage(apitools_messages.Object, name_expansion_result.expanded_result)
    preconditions = Preconditions(gen_match=self.preconditions.gen_match, meta_gen_match=self.preconditions.meta_gen_match)
    if preconditions.gen_match is None:
        preconditions.gen_match = cloud_obj_metadata.generation
    if preconditions.meta_gen_match is None:
        preconditions.meta_gen_match = cloud_obj_metadata.metageneration
    patch_obj_metadata = ObjectMetadataFromHeaders(self.metadata_change)
    api = gsutil_api.GetApiSelector(provider=exp_src_url.scheme)
    if api == ApiSelector.XML:
        pass
    elif api == ApiSelector.JSON:
        CopyObjectMetadata(patch_obj_metadata, cloud_obj_metadata, override=True)
        patch_obj_metadata = cloud_obj_metadata
        patch_obj_metadata.generation = None
        patch_obj_metadata.metageneration = None
    gsutil_api.PatchObjectMetadata(exp_src_url.bucket_name, exp_src_url.object_name, patch_obj_metadata, generation=exp_src_url.generation, preconditions=preconditions, provider=exp_src_url.scheme, fields=['id'])
    _PutToQueueWithTimeout(gsutil_api.status_queue, MetadataMessage(message_time=time.time()))