from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import sys
import six
from gslib.cloud_api import EncryptionException
from gslib.exception import CommandException
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.storage_url import GenerationFromUrlAndString
from gslib.utils.constants import S3_ACL_MARKER_GUID
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import S3_MARKER_GUIDS
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import AclTranslation
from gslib.utils import text_util
from gslib.wildcard_iterator import StorageUrlFromString
Checks bucket listing reference against patterns to exclude.

    Args:
      blr: BucketListingRef to check.

    Returns:
      True if reference matches a pattern and should be excluded.
    