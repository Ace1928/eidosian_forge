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
def _GetHeaderStringsFromSubOpts(self):
    """Gets header values from after the "setmeta" part of the command.

    Example: $ gsutil -h not:parsed setmeta is:parsed gs://bucket/object
               -> ["is:parsed"]

    Returns:
      List[str]: Headers without the "-h" but not yet split on colons.

    Raises:
      CommandException Found canned ACL.
    """
    if not self.sub_opts:
        return []
    headers = []
    for o, a in self.sub_opts:
        if o == '-h':
            if 'x-goog-acl' in a or 'x-amz-acl' in a:
                raise CommandException('gsutil setmeta no longer allows canned ACLs. Use gsutil acl set ... to set canned ACLs.')
            headers.append(a)
    return headers