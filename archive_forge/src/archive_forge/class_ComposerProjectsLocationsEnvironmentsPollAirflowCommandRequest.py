from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsPollAirflowCommandRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsPollAirflowCommandRequest object.

  Fields:
    environment: The resource name of the environment in the form:
      "projects/{projectId}/locations/{locationId}/environments/{environmentId
      }"
    pollAirflowCommandRequest: A PollAirflowCommandRequest resource to be
      passed as the request body.
  """
    environment = _messages.StringField(1, required=True)
    pollAirflowCommandRequest = _messages.MessageField('PollAirflowCommandRequest', 2)