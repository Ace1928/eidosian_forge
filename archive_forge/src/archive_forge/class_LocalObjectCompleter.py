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
class LocalObjectCompleter(object):
    """Completer object for local files."""

    def __init__(self):
        from argcomplete.completers import FilesCompleter
        self.files_completer = FilesCompleter()

    def __call__(self, prefix, **kwargs):
        return self.files_completer(prefix, **kwargs)