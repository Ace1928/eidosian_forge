from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesCreateRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesCreateRequest
  object.

  Fields:
    createMessageRequest: A CreateMessageRequest resource to be passed as the
      request body.
    parent: The name of the HL7v2 store this message belongs to.
  """
    createMessageRequest = _messages.MessageField('CreateMessageRequest', 1)
    parent = _messages.StringField(2, required=True)