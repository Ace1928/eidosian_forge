from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalDeploymentsDevicesGetRequest(_messages.Message):
    """A SasportalDeploymentsDevicesGetRequest object.

  Fields:
    name: Required. The name of the device.
  """
    name = _messages.StringField(1, required=True)