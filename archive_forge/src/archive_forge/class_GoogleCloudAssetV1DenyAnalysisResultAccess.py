from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1DenyAnalysisResultAccess(_messages.Message):
    """An IAM role or permission under analysis.

  Fields:
    permission: The IAM permission in [v1
      format](https://cloud.google.com/iam/docs/permissions-reference)
    role: The IAM role.
  """
    permission = _messages.StringField(1)
    role = _messages.StringField(2)