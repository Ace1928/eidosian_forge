from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1MoveAnalysisResult(_messages.Message):
    """Represents the successful move analysis results for a group.

  Fields:
    blockers: List of blockers. If not resolved, these will result in
      compliance violations in the target.
    warnings: List of warnings. These are risks that may or may not result in
      compliance violations.
  """
    blockers = _messages.MessageField('GoogleCloudAssuredworkloadsV1MoveImpact', 1, repeated=True)
    warnings = _messages.MessageField('GoogleCloudAssuredworkloadsV1MoveImpact', 2, repeated=True)