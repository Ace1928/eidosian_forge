from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudAssetV1p4alpha1Identity(_messages.Message):
    """An identity that appears in an access control list.

  Fields:
    analysisState: The analysis state of this identity node.
    name: The identity name in any form of members appear in [IAM policy
      binding](https://cloud.google.com/iam/reference/rest/v1/Binding), such
      as: - user:foo@google.com - group:group1@google.com -
      serviceAccount:s1@prj1.iam.gserviceaccount.com -
      projectOwner:some_project_id - domain:google.com - allUsers - etc.
  """
    analysisState = _messages.MessageField('GoogleCloudAssetV1p4alpha1AnalysisState', 1)
    name = _messages.StringField(2)