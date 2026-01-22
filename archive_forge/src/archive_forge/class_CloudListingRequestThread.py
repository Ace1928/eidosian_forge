from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import threading
import time
import boto
from boto.gs.acl import CannedACLStrings
from gslib.storage_url import IsFileUrlString
from gslib.storage_url import StorageUrlFromString
from gslib.storage_url import StripOneSlash
from gslib.utils.boto_util import GetTabCompletionCacheFilename
from gslib.utils.boto_util import GetTabCompletionLogFilename
from gslib.wildcard_iterator import CreateWildcardIterator
class CloudListingRequestThread(threading.Thread):
    """Thread that performs a listing request for the given URL string."""

    def __init__(self, wildcard_url_str, gsutil_api):
        """Instantiates Cloud listing request thread.

    Args:
      wildcard_url_str: The URL to list.
      gsutil_api: gsutil Cloud API instance to use.
    """
        super(CloudListingRequestThread, self).__init__()
        self.daemon = True
        self._wildcard_url_str = wildcard_url_str
        self._gsutil_api = gsutil_api
        self.results = None

    def run(self):
        it = CreateWildcardIterator(self._wildcard_url_str, self._gsutil_api).IterAll(bucket_listing_fields=['name'])
        self.results = [str(c) for c in itertools.islice(it, _TAB_COMPLETE_MAX_RESULTS)]