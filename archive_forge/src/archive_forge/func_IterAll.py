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
def IterAll(self, bucket_listing_fields=None, expand_top_level_buckets=False):
    """Iterates over the wildcard, yielding BucketListingRefs.

    Args:
      bucket_listing_fields: Iterable fields to include in listings.
          Ex. ['size']. Currently only 'size' is supported.
          If present, will populate yielded BucketListingObject.root_object
          with the file name and size.
      expand_top_level_buckets: Ignored; filesystems don't have buckets.

    Yields:
      BucketListingRefs of type OBJECT (file) or PREFIX (directory),
      or empty iterator if no matches.
    """
    for bucket_listing_ref in self.__iter__(bucket_listing_fields=bucket_listing_fields):
        yield bucket_listing_ref