from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzerOrgPolicyConstraint(_messages.Message):
    """The organization policy constraint definition.

  Fields:
    customConstraint: The definition of the custom constraint.
    googleDefinedConstraint: The definition of the canned constraint defined
      by Google.
  """
    customConstraint = _messages.MessageField('GoogleCloudAssetV1CustomConstraint', 1)
    googleDefinedConstraint = _messages.MessageField('GoogleCloudAssetV1Constraint', 2)