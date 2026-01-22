from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProvidersNotesDeleteRequest(_messages.Message):
    """A ContaineranalysisProvidersNotesDeleteRequest object.

  Fields:
    name: The name of the note in the form of
      "providers/{provider_id}/notes/{NOTE_ID}"
  """
    name = _messages.StringField(1, required=True)