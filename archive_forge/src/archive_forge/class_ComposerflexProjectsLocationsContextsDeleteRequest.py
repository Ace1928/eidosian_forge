from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerflexProjectsLocationsContextsDeleteRequest(_messages.Message):
    """A ComposerflexProjectsLocationsContextsDeleteRequest object.

  Fields:
    name: The resource name of the context to delete. Must be in the form
      "projects/{projectId}/locations/{locationId}/contexts/{contextId}."
  """
    name = _messages.StringField(1, required=True)