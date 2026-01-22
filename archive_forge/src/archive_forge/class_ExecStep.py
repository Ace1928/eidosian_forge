from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ExecStep(_messages.Message):
    """A step that runs an executable for a PatchJob.

  Fields:
    linuxExecStepConfig: The ExecStepConfig for all Linux VMs targeted by the
      PatchJob.
    windowsExecStepConfig: The ExecStepConfig for all Windows VMs targeted by
      the PatchJob.
  """
    linuxExecStepConfig = _messages.MessageField('ExecStepConfig', 1)
    windowsExecStepConfig = _messages.MessageField('ExecStepConfig', 2)