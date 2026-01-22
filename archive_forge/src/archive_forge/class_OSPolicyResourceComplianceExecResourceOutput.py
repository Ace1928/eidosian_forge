from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourceComplianceExecResourceOutput(_messages.Message):
    """ExecResource specific output.

  Fields:
    enforcementOutput: Output from Enforcement phase output file (if run).
      Output size is limited to 100K bytes.
  """
    enforcementOutput = _messages.BytesField(1)