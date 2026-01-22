from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsRepairEventingRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsRepairEventingRequest object.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/*/connections/*`
    repairEventingRequest: A RepairEventingRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    repairEventingRequest = _messages.MessageField('RepairEventingRequest', 2)