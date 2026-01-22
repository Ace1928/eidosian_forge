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
def MakeCompleter(completer_type, gsutil_api):
    """Create a completer instance of the given type.

  Args:
    completer_type: The type of completer to create.
    gsutil_api: gsutil Cloud API instance to use.
  Returns:
    A completer instance.
  Raises:
    RuntimeError: if completer type is not supported.
  """
    if completer_type == CompleterType.CLOUD_OR_LOCAL_OBJECT:
        return CloudOrLocalObjectCompleter(gsutil_api)
    elif completer_type == CompleterType.LOCAL_OBJECT:
        return LocalObjectCompleter()
    elif completer_type == CompleterType.LOCAL_OBJECT_OR_CANNED_ACL:
        return LocalObjectOrCannedACLCompleter()
    elif completer_type == CompleterType.CLOUD_BUCKET:
        return CloudObjectCompleter(gsutil_api, bucket_only=True)
    elif completer_type == CompleterType.CLOUD_OBJECT:
        return CloudObjectCompleter(gsutil_api)
    elif completer_type == CompleterType.NO_OP:
        return NoOpCompleter()
    else:
        raise RuntimeError('Unknown completer "%s"' % completer_type)