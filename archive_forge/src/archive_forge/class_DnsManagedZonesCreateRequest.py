from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class DnsManagedZonesCreateRequest(_messages.Message):
    """A DnsManagedZonesCreateRequest object.

  Fields:
    managedZone: A ManagedZone resource to be passed as the request body.
    project: Identifies the project addressed by this request.
  """
    managedZone = _messages.MessageField('ManagedZone', 1)
    project = _messages.StringField(2, required=True)