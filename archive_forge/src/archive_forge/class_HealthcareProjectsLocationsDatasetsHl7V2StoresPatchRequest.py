from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresPatchRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresPatchRequest object.

  Fields:
    hl7V2Store: A Hl7V2Store resource to be passed as the request body.
    name: Identifier. Resource name of the HL7v2 store, of the form `projects/
      {project_id}/locations/{location_id}/datasets/{dataset_id}/hl7V2Stores/{
      hl7v2_store_id}`.
    updateMask: Required. The update mask applies to the resource. For the
      `FieldMask` definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask
  """
    hl7V2Store = _messages.MessageField('Hl7V2Store', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)