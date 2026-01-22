from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalDatasetReference(_messages.Message):
    """Configures the access a dataset defined in an external metadata storage.

  Fields:
    connection: Required. The connection id that is used to access the
      external_source. Format: projects/{project_id}/locations/{location_id}/c
      onnections/{connection_id}
    externalSource: Required. External source that backs this dataset.
  """
    connection = _messages.StringField(1)
    externalSource = _messages.StringField(2)