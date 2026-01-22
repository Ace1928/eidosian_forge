from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1KeyValueEntry(_messages.Message):
    """Key value map pair where the value represents the data associated with
  the corresponding key. **Note**: Supported for Apigee hybrid 1.8.x and
  higher.

  Fields:
    name: Resource URI that can be used to identify the scope of the key value
      map entries.
    value: Required. Data or payload that is being retrieved and associated
      with the unique key.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)