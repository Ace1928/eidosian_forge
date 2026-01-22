from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsLoadSnapshotRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsLoadSnapshotRequest object.

  Fields:
    environment: The resource name of the target environment in the form:
      "projects/{projectId}/locations/{locationId}/environments/{environmentId
      }"
    loadSnapshotRequest: A LoadSnapshotRequest resource to be passed as the
      request body.
  """
    environment = _messages.StringField(1, required=True)
    loadSnapshotRequest = _messages.MessageField('LoadSnapshotRequest', 2)