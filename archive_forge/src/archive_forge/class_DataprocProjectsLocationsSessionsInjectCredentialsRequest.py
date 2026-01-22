from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsSessionsInjectCredentialsRequest(_messages.Message):
    """A DataprocProjectsLocationsSessionsInjectCredentialsRequest object.

  Fields:
    injectSessionCredentialsRequest: A InjectSessionCredentialsRequest
      resource to be passed as the request body.
    session: Required. The name of the session resource to inject credentials
      to.
  """
    injectSessionCredentialsRequest = _messages.MessageField('InjectSessionCredentialsRequest', 1)
    session = _messages.StringField(2, required=True)