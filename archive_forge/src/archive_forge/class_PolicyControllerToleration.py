from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyControllerToleration(_messages.Message):
    """Toleration of a node taint.

  Fields:
    effect: Matches a taint effect.
    key: Matches a taint key (not necessarily unique).
    operator: Matches a taint operator.
    value: Matches a taint value.
  """
    effect = _messages.StringField(1)
    key = _messages.StringField(2)
    operator = _messages.StringField(3)
    value = _messages.StringField(4)