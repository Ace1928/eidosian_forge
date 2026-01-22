from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesDevicesDeleteRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesDevicesDeleteRequest object.

  Fields:
    name: Required. The name of the device. For example,
      `projects/p0/locations/us-central1/registries/registry0/devices/device0`
      or `projects/p0/locations/us-
      central1/registries/registry0/devices/{num_id}`.
  """
    name = _messages.StringField(1, required=True)