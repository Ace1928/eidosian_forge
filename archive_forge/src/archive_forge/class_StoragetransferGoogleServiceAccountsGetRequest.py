from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferGoogleServiceAccountsGetRequest(_messages.Message):
    """A StoragetransferGoogleServiceAccountsGetRequest object.

  Fields:
    projectId: Required. The ID of the Google Cloud project that the Google
      service account is associated with.
  """
    projectId = _messages.StringField(1, required=True)