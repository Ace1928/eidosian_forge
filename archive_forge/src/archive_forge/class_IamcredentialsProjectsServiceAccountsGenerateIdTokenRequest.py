from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamcredentialsProjectsServiceAccountsGenerateIdTokenRequest(_messages.Message):
    """A IamcredentialsProjectsServiceAccountsGenerateIdTokenRequest object.

  Fields:
    generateIdTokenRequest: A GenerateIdTokenRequest resource to be passed as
      the request body.
    name: The resource name of the service account for which the credentials
      are requested, in the following format:
      `projects/-/serviceAccounts/{ACCOUNT_EMAIL_OR_UNIQUEID}`. The `-`
      wildcard character is required; replacing it with a project ID is
      invalid.
  """
    generateIdTokenRequest = _messages.MessageField('GenerateIdTokenRequest', 1)
    name = _messages.StringField(2, required=True)