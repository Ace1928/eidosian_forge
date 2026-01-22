from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamcredentialsProjectsServiceAccountsGenerateAccessTokenRequest(_messages.Message):
    """A IamcredentialsProjectsServiceAccountsGenerateAccessTokenRequest
  object.

  Fields:
    generateAccessTokenRequest: A GenerateAccessTokenRequest resource to be
      passed as the request body.
    name: The resource name of the service account for which the credentials
      are requested, in the following format:
      `projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}`. The `-`
      wildcard character is required; replacing it with a project ID is
      invalid.
  """
    generateAccessTokenRequest = _messages.MessageField('GenerateAccessTokenRequest', 1)
    name = _messages.StringField(2, required=True)