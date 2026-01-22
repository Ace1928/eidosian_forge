from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import glob
import logging
import os
import re
import textwrap
import six
from gslib.bucket_listing_ref import BucketListingBucket
from gslib.bucket_listing_ref import BucketListingObject
from gslib.bucket_listing_ref import BucketListingPrefix
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import NotFoundException
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import StripOneSlash
from gslib.storage_url import WILDCARD_REGEX
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import UTF8
from gslib.utils.text_util import FixWindowsEncodingIfNeeded
from gslib.utils.text_util import PrintableStr
def _GetToListFields(self, get_fields=None):
    """Prepends 'items/' to the input fields and converts it to a set.

    This way field sets requested for GetBucket can be used in ListBucket calls.
    Note that the input set must contain only bucket or object fields; listing
    fields such as prefixes or nextPageToken should be added after calling
    this function.

    Args:
      get_fields: Iterable fields usable in GetBucket/GetObject calls.

    Returns:
      Set of fields usable in ListBuckets/ListObjects calls.
    """
    if get_fields:
        list_fields = set()
        for field in get_fields:
            list_fields.add('items/' + field)
        return list_fields