from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalCustomersDeploymentsDevicesCreateSignedRequest(_messages.Message):
    """A SasportalCustomersDeploymentsDevicesCreateSignedRequest object.

  Fields:
    parent: Required. The name of the parent resource.
    sasPortalCreateSignedDeviceRequest: A SasPortalCreateSignedDeviceRequest
      resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    sasPortalCreateSignedDeviceRequest = _messages.MessageField('SasPortalCreateSignedDeviceRequest', 2)