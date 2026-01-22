from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesDeleteRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesDeleteRequest
  object.

  Fields:
    name: The resource name of the HL7v2 message to delete.
  """
    name = _messages.StringField(1, required=True)