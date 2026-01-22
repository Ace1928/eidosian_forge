from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsDeleteRequest(_messages.Message):
    """A ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsDeleteRequest
  object.

  Fields:
    name: Required. The Secret to delete, in the form: "projects/{projectId}/l
      ocations/{locationId}/environments/{environmentId}/userWorkloadsSecrets/
      {userWorkloadsSecretId}"
  """
    name = _messages.StringField(1, required=True)