from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsZonesInstancesLookupConfigsRequest(_messages.Message):
    """A OsconfigProjectsZonesInstancesLookupConfigsRequest object.

  Fields:
    lookupConfigsRequest: A LookupConfigsRequest resource to be passed as the
      request body.
    resource: The resource name for the instance.
  """
    lookupConfigsRequest = _messages.MessageField('LookupConfigsRequest', 1)
    resource = _messages.StringField(2, required=True)