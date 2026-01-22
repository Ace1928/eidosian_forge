from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnywhereCache(_messages.Message):
    """An Anywhere Cache instance.

  Fields:
    admissionPolicy: The cache-level entry admission policy.
    anywhereCacheId: The ID of the Anywhere cache instance.
    bucket: The name of the bucket containing this cache instance.
    createTime: The creation time of the cache instance in RFC 3339 format.
    id: The ID of the resource, including the project number, bucket name and
      anywhere cache ID.
    kind: The kind of item this is. For Anywhere Cache, this is always
      storage#anywhereCache.
    pendingUpdate: True if the cache instance has an active Update long-
      running operation.
    selfLink: The link to this cache instance.
    state: The current state of the cache instance.
    ttl: The TTL of all cache entries in whole seconds. e.g., "7200s".
    updateTime: The modification time of the cache instance metadata in RFC
      3339 format.
    zone: The zone in which the cache instance is running. For example, us-
      central1-a.
  """
    admissionPolicy = _messages.StringField(1)
    anywhereCacheId = _messages.StringField(2)
    bucket = _messages.StringField(3)
    createTime = _message_types.DateTimeField(4)
    id = _messages.StringField(5)
    kind = _messages.StringField(6, default='storage#anywhereCache')
    pendingUpdate = _messages.BooleanField(7)
    selfLink = _messages.StringField(8)
    state = _messages.StringField(9)
    ttl = _messages.StringField(10)
    updateTime = _message_types.DateTimeField(11)
    zone = _messages.StringField(12)