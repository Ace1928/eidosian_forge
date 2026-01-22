from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplianceStandard(_messages.Message):
    """A ComplianceStandard object.

  Fields:
    standard: Name of the compliance standard.
  """
    standard = _messages.StringField(1)