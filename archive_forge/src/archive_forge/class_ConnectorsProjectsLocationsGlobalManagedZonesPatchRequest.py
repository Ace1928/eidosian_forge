from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsGlobalManagedZonesPatchRequest(_messages.Message):
    """A ConnectorsProjectsLocationsGlobalManagedZonesPatchRequest object.

  Fields:
    managedZone: A ManagedZone resource to be passed as the request body.
    name: Output only. Resource name of the Managed Zone. Format:
      projects/{project}/locations/global/managedZones/{managed_zone}
    updateMask: Required. The list of fields to update. Fields are specified
      relative to the managedZone. A field will be overwritten if it is in the
      mask. You can modify only the fields listed below. To update the
      managedZone details: * `description` * `labels` * `target_project` *
      `target_network`
  """
    managedZone = _messages.MessageField('ManagedZone', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)