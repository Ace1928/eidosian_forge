from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsResetAuthorizationCodeRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsResetAuthorizationCodeRequest
  object.

  Fields:
    registration: Required. The name of the `Registration` whose authorization
      code is being reset, in the format
      `projects/*/locations/*/registrations/*`.
    resetAuthorizationCodeRequest: A ResetAuthorizationCodeRequest resource to
      be passed as the request body.
  """
    registration = _messages.StringField(1, required=True)
    resetAuthorizationCodeRequest = _messages.MessageField('ResetAuthorizationCodeRequest', 2)