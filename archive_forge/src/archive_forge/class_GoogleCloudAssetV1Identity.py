from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1Identity(_messages.Message):
    """An identity under analysis.

  Fields:
    analysisState: The analysis state of this identity.
    name: The identity of members, formatted as appear in an [IAM policy
      binding](https://cloud.google.com/iam/reference/rest/v1/Binding). For
      example, they might be formatted like the following: -
      user:foo@google.com - group:group1@google.com -
      serviceAccount:s1@prj1.iam.gserviceaccount.com -
      projectOwner:some_project_id - domain:google.com - allUsers
  """
    analysisState = _messages.MessageField('IamPolicyAnalysisState', 1)
    name = _messages.StringField(2)