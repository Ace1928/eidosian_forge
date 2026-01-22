from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalNodesDevicesUpdateSignedRequest(_messages.Message):
    """A SasportalNodesDevicesUpdateSignedRequest object.

  Fields:
    name: Required. The name of the device to update.
    sasPortalUpdateSignedDeviceRequest: A SasPortalUpdateSignedDeviceRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    sasPortalUpdateSignedDeviceRequest = _messages.MessageField('SasPortalUpdateSignedDeviceRequest', 2)