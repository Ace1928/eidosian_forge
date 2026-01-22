from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudiotProjectsLocationsRegistriesCreateRequest(_messages.Message):
    """A CloudiotProjectsLocationsRegistriesCreateRequest object.

  Fields:
    deviceRegistry: A DeviceRegistry resource to be passed as the request
      body.
    parent: Required. The project and cloud region where this device registry
      must be created. For example, `projects/example-project/locations/us-
      central1`.
  """
    deviceRegistry = _messages.MessageField('DeviceRegistry', 1)
    parent = _messages.StringField(2, required=True)