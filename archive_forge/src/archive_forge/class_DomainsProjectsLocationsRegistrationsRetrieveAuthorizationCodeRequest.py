from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsRetrieveAuthorizationCodeRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsRetrieveAuthorizationCodeRequest
  object.

  Fields:
    registration: Required. The name of the `Registration` whose authorization
      code is being retrieved, in the format
      `projects/*/locations/*/registrations/*`.
  """
    registration = _messages.StringField(1, required=True)