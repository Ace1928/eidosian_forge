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
def _WriteTimingLog(message):
    """Write an entry to the tab completion timing log, if it's enabled."""
    if boto.config.getbool('GSUtil', 'tab_completion_time_logs', False):
        with open(GetTabCompletionLogFilename(), 'ab') as fp:
            fp.write(message)