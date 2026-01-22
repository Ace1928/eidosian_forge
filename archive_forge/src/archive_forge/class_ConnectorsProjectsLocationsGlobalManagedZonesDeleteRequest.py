from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsGlobalManagedZonesDeleteRequest(_messages.Message):
    """A ConnectorsProjectsLocationsGlobalManagedZonesDeleteRequest object.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/global/managedZones/*`
  """
    name = _messages.StringField(1, required=True)