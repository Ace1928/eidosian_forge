from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CrossConnectItem(_messages.Message):
    """Item for controlling single cross connect network's configuration.

  Fields:
    displayName: Optional. display_name is intended only for UI elements to
      help humans identify this item, not as a unique identifier.
    privateEndpoint: Output only. The internal IP address of this cluster's
      endpoint in the requested subnetwork.
    subnetwork: Subnetworks where cluster's private endpoint is accessible.
      specified in projects/*/regions/*/subnetworks/* format.
  """
    displayName = _messages.StringField(1)
    privateEndpoint = _messages.StringField(2)
    subnetwork = _messages.StringField(3)