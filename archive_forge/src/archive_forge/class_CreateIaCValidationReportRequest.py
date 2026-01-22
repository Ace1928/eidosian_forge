from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateIaCValidationReportRequest(_messages.Message):
    """Request message for creating an IaC validation report.

  Fields:
    iac: Required. The infrastrucutre as code which is to be validated for
      generating report.
  """
    iac = _messages.MessageField('IaC', 1)