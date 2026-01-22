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
def _IterDir(self, directory, wildcard):
    """An iterator over the specified dir and wildcard.

    Args:
      directory (unicode): The path of the directory to iterate over.
      wildcard (str): The wildcard characters used for filename pattern
          matching.

    Yields:
      (str) A string containing the path to a file somewhere under the directory
      hierarchy of `directory`.

    Raises:
      ComandException: If this method encounters a file path that it cannot
      decode as UTF-8.
    """
    if os.path.splitdrive(directory)[0] == directory:
        directory += '\\'
    for dirpath, dirnames, filenames in os.walk(six.ensure_text(directory), topdown=True):
        filtered_dirnames = []
        for dirname in dirnames:
            full_dir_path = os.path.join(dirpath, dirname)
            if not self._ExcludeDir(full_dir_path):
                filtered_dirnames.append(dirname)
            else:
                continue
            if self.logger and os.path.islink(full_dir_path):
                self.logger.info('Skipping symlink directory "%s"', full_dir_path)
        dirnames[:] = filtered_dirnames
        for f in fnmatch.filter(filenames, wildcard):
            try:
                yield os.path.join(dirpath, FixWindowsEncodingIfNeeded(f))
            except UnicodeDecodeError:
                raise CommandException('\n'.join(textwrap.wrap(_UNICODE_EXCEPTION_TEXT % repr(os.path.join(dirpath, f)))))