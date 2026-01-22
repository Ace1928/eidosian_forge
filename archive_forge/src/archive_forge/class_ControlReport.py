from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ControlReport(_messages.Message):
    """Represents a control audit report.

  Fields:
    controlDetails: Output only. Control details for the report including the
      findings.
    name: Identifier. The name of this Control Report, in the format of scope
      given in request.
  """
    controlDetails = _messages.MessageField('ControlDetails', 1)
    name = _messages.StringField(2)