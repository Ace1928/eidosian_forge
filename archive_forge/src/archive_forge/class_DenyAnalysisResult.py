from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DenyAnalysisResult(_messages.Message):
    """A deny policy analysis result for an access tuple.

  Fields:
    accessTuple: An access tuple that is conducted deny policy analysis. This
      access tuple should match at least one access tuple derived from
      IamPolicyAnalysisResult.
    denyDetails: The details about how denied_access_tuple is denied. If it is
      empty, it means no deny rule is found to have any effect on the access
      tuple.
  """
    accessTuple = _messages.MessageField('GoogleCloudAssetV1DenyAnalysisResultAccessTuple', 1)
    denyDetails = _messages.MessageField('GoogleCloudAssetV1DenyAnalysisResultDenyDetail', 2, repeated=True)