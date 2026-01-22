from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1KeyValueMap(_messages.Message):
    """Collection of key/value string pairs.

  Fields:
    encrypted: Required. Flag that specifies whether entry values will be
      encrypted. This field is retained for backward compatibility and the
      value of encrypted will always be `true`. Apigee X and hybrid do not
      support unencrypted key value maps.
    name: Required. ID of the key value map.
    resourceName: Output only. Resource URI on which the key value map is
      based.
  """
    encrypted = _messages.BooleanField(1)
    name = _messages.StringField(2)
    resourceName = _messages.StringField(3)