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
def CreateWildcardIterator(url_str, gsutil_api, all_versions=False, project_id=None, exclude_tuple=None, ignore_symlinks=False, logger=None):
    """Instantiate a WildcardIterator for the given URL string.

  Args:
    url_str: URL string naming wildcard object(s) to iterate.
    gsutil_api: Cloud storage interface.  Passed in for thread safety, also
                settable for testing/mocking.
    all_versions: If true, the iterator yields all versions of objects
                  matching the wildcard.  If false, yields just the live
                  object version.
    project_id: Project id to use for bucket listings.
    exclude_tuple: (base_url, exclude_pattern), where base_url is
                   top-level URL to list; exclude_pattern is a regex
                   of paths to ignore during iteration.
    ignore_symlinks: For FileUrls, ignore symlinks during iteration if true.
    logger: logging.Logger used for outputting debug messages during iteration.
            If None, the root logger will be used.

  Returns:
    A WildcardIterator that handles the requested iteration.
  """
    url = StorageUrlFromString(url_str)
    logger = logger or logging.getLogger()
    if url.IsFileUrl():
        return FileWildcardIterator(url, exclude_tuple=exclude_tuple, ignore_symlinks=ignore_symlinks, logger=logger)
    else:
        return CloudWildcardIterator(url, gsutil_api, all_versions=all_versions, project_id=project_id)