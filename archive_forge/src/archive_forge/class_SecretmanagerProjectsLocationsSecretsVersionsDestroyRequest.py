from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecretmanagerProjectsLocationsSecretsVersionsDestroyRequest(_messages.Message):
    """A SecretmanagerProjectsLocationsSecretsVersionsDestroyRequest object.

  Fields:
    destroySecretVersionRequest: A DestroySecretVersionRequest resource to be
      passed as the request body.
    name: Required. The resource name of the SecretVersion to destroy in the
      format `projects/*/secrets/*/versions/*` or
      `projects/*/locations/*/secrets/*/versions/*`.
  """
    destroySecretVersionRequest = _messages.MessageField('DestroySecretVersionRequest', 1)
    name = _messages.StringField(2, required=True)