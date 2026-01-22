from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesDevicesModifyCloudToDeviceConfigRequest(_messages.Message):
    """A
  CloudiotProjectsLocationsRegistriesDevicesModifyCloudToDeviceConfigRequest
  object.

  Fields:
    modifyCloudToDeviceConfigRequest: A ModifyCloudToDeviceConfigRequest
      resource to be passed as the request body.
    name: Required. The name of the device. For example,
      `projects/p0/locations/us-central1/registries/registry0/devices/device0`
      or `projects/p0/locations/us-
      central1/registries/registry0/devices/{num_id}`.
  """
    modifyCloudToDeviceConfigRequest = _messages.MessageField('ModifyCloudToDeviceConfigRequest', 1)
    name = _messages.StringField(2, required=True)