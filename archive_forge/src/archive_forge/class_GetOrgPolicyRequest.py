from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GetOrgPolicyRequest(_messages.Message):
    """The request sent to the GetOrgPolicy method.

  Fields:
    constraint: Name of the `Constraint` to get the `Policy`.
  """
    constraint = _messages.StringField(1)