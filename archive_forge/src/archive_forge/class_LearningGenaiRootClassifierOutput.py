from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootClassifierOutput(_messages.Message):
    """A LearningGenaiRootClassifierOutput object.

  Fields:
    ruleOutput: If set, this is the output of the first matching rule.
    ruleOutputs: outputs of all matching rule.
    state: The results of data_providers and metrics.
  """
    ruleOutput = _messages.MessageField('LearningGenaiRootRuleOutput', 1)
    ruleOutputs = _messages.MessageField('LearningGenaiRootRuleOutput', 2, repeated=True)
    state = _messages.MessageField('LearningGenaiRootClassifierState', 3)