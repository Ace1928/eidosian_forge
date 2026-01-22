from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPolicy(_messages.Message):
    """Cloud IAM Policy information associated with the Google Cloud resource
  described by the Security Command Center asset. This information is managed
  and defined by the Google Cloud resource and cannot be modified by the user.

  Fields:
    policyBlob: The JSON representation of the Policy associated with the
      asset. See https://cloud.google.com/iam/reference/rest/v1/Policy for
      format details.
  """
    policyBlob = _messages.StringField(1)