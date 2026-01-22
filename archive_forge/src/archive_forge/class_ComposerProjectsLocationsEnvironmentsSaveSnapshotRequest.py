from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsSaveSnapshotRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsSaveSnapshotRequest object.

  Fields:
    environment: The resource name of the source environment in the form:
      "projects/{projectId}/locations/{locationId}/environments/{environmentId
      }"
    saveSnapshotRequest: A SaveSnapshotRequest resource to be passed as the
      request body.
  """
    environment = _messages.StringField(1, required=True)
    saveSnapshotRequest = _messages.MessageField('SaveSnapshotRequest', 2)