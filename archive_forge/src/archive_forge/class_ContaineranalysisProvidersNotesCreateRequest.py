from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProvidersNotesCreateRequest(_messages.Message):
    """A ContaineranalysisProvidersNotesCreateRequest object.

  Fields:
    name: The name of the project. Should be of the form
      "providers/{provider_id}". @Deprecated
    note: A Note resource to be passed as the request body.
    noteId: The ID to use for this note.
    parent: This field contains the project Id for example:
      "projects/{project_id}
  """
    name = _messages.StringField(1, required=True)
    note = _messages.MessageField('Note', 2)
    noteId = _messages.StringField(3)
    parent = _messages.StringField(4)