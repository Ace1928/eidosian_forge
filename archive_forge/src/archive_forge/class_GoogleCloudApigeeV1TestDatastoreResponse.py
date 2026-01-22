from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1TestDatastoreResponse(_messages.Message):
    """The response for TestDatastore

  Fields:
    error: Output only. Error message of test connection failure
    state: Output only. It could be `completed` or `failed`
  """
    error = _messages.StringField(1)
    state = _messages.StringField(2)