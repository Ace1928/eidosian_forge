from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesDevicesPatchRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesDevicesPatchRequest object.

  Fields:
    device: A Device resource to be passed as the request body.
    name: The resource path name. For example, `projects/p1/locations/us-
      central1/registries/registry0/devices/dev0` or
      `projects/p1/locations/us-
      central1/registries/registry0/devices/{num_id}`. When `name` is
      populated as a response from the service, it always ends in the device
      numeric ID.
    updateMask: Required. Only updates the `device` fields indicated by this
      mask. The field mask must not be empty, and it must not contain fields
      that are immutable or only set by the server. Mutable top-level fields:
      `credentials`, `blocked`, and `metadata`
  """
    device = _messages.MessageField('Device', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)