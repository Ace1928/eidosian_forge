from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsEntryTypesPatchRequest(_messages.Message):
    """A DataplexProjectsLocationsEntryTypesPatchRequest object.

  Fields:
    googleCloudDataplexV1EntryType: A GoogleCloudDataplexV1EntryType resource
      to be passed as the request body.
    name: Output only. The relative resource name of the EntryType, of the
      form: projects/{project_number}/locations/{location_id}/entryTypes/{entr
      y_type_id}.
    updateMask: Required. Mask of fields to update.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1EntryType = _messages.MessageField('GoogleCloudDataplexV1EntryType', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)