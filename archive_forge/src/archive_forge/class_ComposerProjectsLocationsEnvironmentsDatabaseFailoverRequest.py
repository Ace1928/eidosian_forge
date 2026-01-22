from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsDatabaseFailoverRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsDatabaseFailoverRequest object.

  Fields:
    databaseFailoverRequest: A DatabaseFailoverRequest resource to be passed
      as the request body.
    environment: Target environment:
      "projects/{projectId}/locations/{locationId}/environments/{environmentId
      }"
  """
    databaseFailoverRequest = _messages.MessageField('DatabaseFailoverRequest', 1)
    environment = _messages.StringField(2, required=True)