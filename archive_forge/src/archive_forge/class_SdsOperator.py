from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SdsOperator(_messages.Message):
    """Config for the SDS Operator add-on which installs Robin CNS.

  Fields:
    version: Optional. SDS Operator version.
  """
    version = _messages.StringField(1)