from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicenetworkingServicesDnsZonesRemoveRequest(_messages.Message):
    """A ServicenetworkingServicesDnsZonesRemoveRequest object.

  Fields:
    parent: Required. The service that is managing peering connectivity for a
      service producer's organization. For Google services that support this
      functionality, this value is
      `services/servicenetworking.googleapis.com`.
    removeDnsZoneRequest: A RemoveDnsZoneRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    removeDnsZoneRequest = _messages.MessageField('RemoveDnsZoneRequest', 2)