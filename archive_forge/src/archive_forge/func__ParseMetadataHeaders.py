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
def _ParseMetadataHeaders(self, headers):
    """Validates and parses metadata changes from the headers argument.

    Args:
      headers: Header dict to validate and parse.

    Returns:
      (metadata_plus, metadata_minus): Tuple of header sets to add and remove.
    """
    metadata_minus = set()
    cust_metadata_minus = set()
    metadata_plus = {}
    cust_metadata_plus = {}
    num_metadata_plus_elems = 0
    num_cust_metadata_plus_elems = 0
    num_metadata_minus_elems = 0
    num_cust_metadata_minus_elems = 0
    for md_arg in headers:
        parts = md_arg.partition(':')
        header, _, value = parts
        InsistAsciiHeader(header)
        lowercase_header = header.lower()
        is_custom_meta = IsCustomMetadataHeader(lowercase_header)
        if not is_custom_meta and lowercase_header not in SETTABLE_FIELDS:
            raise CommandException('Invalid or disallowed header (%s).\nOnly these fields (plus x-goog-meta-* fields) can be set or unset:\n%s' % (header, sorted(list(SETTABLE_FIELDS))))
        if value:
            if is_custom_meta:
                cust_metadata_plus[header] = value
                num_cust_metadata_plus_elems += 1
            else:
                InsistAsciiHeaderValue(header, value)
                value = str(value)
                metadata_plus[lowercase_header] = value
                num_metadata_plus_elems += 1
        elif is_custom_meta:
            cust_metadata_minus.add(header)
            num_cust_metadata_minus_elems += 1
        else:
            metadata_minus.add(lowercase_header)
            num_metadata_minus_elems += 1
    if num_metadata_plus_elems != len(metadata_plus) or num_cust_metadata_plus_elems != len(cust_metadata_plus) or num_metadata_minus_elems != len(metadata_minus) or (num_cust_metadata_minus_elems != len(cust_metadata_minus)) or metadata_minus.intersection(set(metadata_plus.keys())):
        raise CommandException('Each header must appear at most once.')
    metadata_plus.update(cust_metadata_plus)
    metadata_minus.update(cust_metadata_minus)
    return (metadata_minus, metadata_plus)