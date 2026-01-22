from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IamProjectsServiceAccountsKeysCreateRequest(_messages.Message):
    """A IamProjectsServiceAccountsKeysCreateRequest object.

  Fields:
    createServiceAccountKeyRequest: A CreateServiceAccountKeyRequest resource
      to be passed as the request body.
    name: The resource name of the service account in the following format:
      `projects/{project}/serviceAccounts/{account}`. Using `-` as a wildcard
      for the project will infer the project from the account. The `account`
      value can be the `email` address or the `unique_id` of the service
      account.
  """
    createServiceAccountKeyRequest = _messages.MessageField('CreateServiceAccountKeyRequest', 1)
    name = _messages.StringField(2, required=True)