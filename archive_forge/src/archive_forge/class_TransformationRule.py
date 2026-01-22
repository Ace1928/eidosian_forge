from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransformationRule(_messages.Message):
    """A transformation rule to be applied against Kubernetes resources as they
  are selected for restoration from a Backup. A rule contains both filtering
  logic (which resources are subject to transform) and transformation logic.

  Fields:
    description: Optional. The description is a user specified string
      description of the transformation rule.
    fieldActions: Required. A list of transformation rule actions to take
      against candidate resources. Actions are executed in order defined -
      this order matters, as they could potentially interfere with each other
      and the first operation could affect the outcome of the second
      operation.
    resourceFilter: Optional. This field is used to specify a set of fields
      that should be used to determine which resources in backup should be
      acted upon by the supplied transformation rule actions, and this will
      ensure that only specific resources are affected by transformation rule
      actions.
  """
    description = _messages.StringField(1)
    fieldActions = _messages.MessageField('TransformationRuleAction', 2, repeated=True)
    resourceFilter = _messages.MessageField('ResourceFilter', 3)