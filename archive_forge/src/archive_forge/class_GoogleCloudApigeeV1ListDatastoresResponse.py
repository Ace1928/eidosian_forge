from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListDatastoresResponse(_messages.Message):
    """The response for ListDatastores

  Fields:
    datastores: A list of datastores
  """
    datastores = _messages.MessageField('GoogleCloudApigeeV1Datastore', 1, repeated=True)