from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MoveAnalysis(_messages.Message):
    """A message to group the analysis information.

  Fields:
    analysis: Analysis result of moving the target resource.
    displayName: The user friendly display name of the analysis. E.g. IAM,
      organization policy etc.
    error: Description of error encountered when performing the analysis.
  """
    analysis = _messages.MessageField('MoveAnalysisResult', 1)
    displayName = _messages.StringField(2)
    error = _messages.MessageField('Status', 3)