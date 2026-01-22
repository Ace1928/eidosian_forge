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
class LocalObjectOrCannedACLCompleter(object):
    """Completer object for local files and canned ACLs.

  Currently, only Google Cloud Storage canned ACL names are supported.
  """

    def __init__(self):
        self.local_object_completer = LocalObjectCompleter()

    def __call__(self, prefix, **kwargs):
        local_objects = self.local_object_completer(prefix, **kwargs)
        canned_acls = [acl for acl in CannedACLStrings if acl.startswith(prefix)]
        return local_objects + canned_acls