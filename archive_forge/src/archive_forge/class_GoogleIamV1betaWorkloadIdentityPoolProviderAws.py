from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV1betaWorkloadIdentityPoolProviderAws(_messages.Message):
    """Represents an Amazon Web Services identity provider.

  Fields:
    accountId: Required. The AWS account ID.
  """
    accountId = _messages.StringField(1)