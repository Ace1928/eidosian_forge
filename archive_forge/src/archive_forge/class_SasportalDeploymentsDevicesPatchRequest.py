from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalDeploymentsDevicesPatchRequest(_messages.Message):
    """A SasportalDeploymentsDevicesPatchRequest object.

  Fields:
    name: Output only. The resource path name.
    sasPortalDevice: A SasPortalDevice resource to be passed as the request
      body.
    updateMask: Fields to be updated.
  """
    name = _messages.StringField(1, required=True)
    sasPortalDevice = _messages.MessageField('SasPortalDevice', 2)
    updateMask = _messages.StringField(3)