from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesPatchRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesPatchRequest
  object.

  Fields:
    message: A Message resource to be passed as the request body.
    name: Resource name of the Message, of the form `projects/{project_id}/loc
      ations/{location_id}/datasets/{dataset_id}/hl7V2Stores/{hl7_v2_store_id}
      /messages/{message_id}`. Assigned by the server.
    updateMask: The update mask applies to the resource. For the `FieldMask`
      definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    message = _messages.MessageField('Message', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)