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
class CompleterType(object):
    CLOUD_BUCKET = 'cloud_bucket'
    CLOUD_OBJECT = 'cloud_object'
    CLOUD_OR_LOCAL_OBJECT = 'cloud_or_local_object'
    LOCAL_OBJECT = 'local_object'
    LOCAL_OBJECT_OR_CANNED_ACL = 'local_object_or_canned_acl'
    NO_OP = 'no_op'