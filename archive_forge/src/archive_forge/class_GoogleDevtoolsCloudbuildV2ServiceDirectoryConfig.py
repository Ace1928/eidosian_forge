from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV2ServiceDirectoryConfig(_messages.Message):
    """ServiceDirectoryConfig represents Service Directory configuration for a
  connection.

  Fields:
    service: Required. The Service Directory service name. Format: projects/{p
      roject}/locations/{location}/namespaces/{namespace}/services/{service}.
  """
    service = _messages.StringField(1)