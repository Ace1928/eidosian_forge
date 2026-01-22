from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CutoverStep(_messages.Message):
    """CutoverStep holds information about the cutover step progress.

  Fields:
    endTime: The time the step has ended.
    finalSync: Final sync step.
    instantiatingMigratedVm: Instantiating migrated VM step.
    preparingVmDisks: Preparing VM disks step.
    previousReplicationCycle: A replication cycle prior cutover step.
    shuttingDownSourceVm: Shutting down VM step.
    startTime: The time the step has started.
  """
    endTime = _messages.StringField(1)
    finalSync = _messages.MessageField('ReplicationCycle', 2)
    instantiatingMigratedVm = _messages.MessageField('InstantiatingMigratedVMStep', 3)
    preparingVmDisks = _messages.MessageField('PreparingVMDisksStep', 4)
    previousReplicationCycle = _messages.MessageField('ReplicationCycle', 5)
    shuttingDownSourceVm = _messages.MessageField('ShuttingDownSourceVMStep', 6)
    startTime = _messages.StringField(7)