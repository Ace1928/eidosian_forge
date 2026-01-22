from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchedulePolicy(_messages.Message):
    """A policy for scheduling replications.

  Fields:
    idleDuration: The idle duration between replication stages.
    skipOsAdaptation: A flag to indicate whether to skip OS adaptation during
      the replication sync. OS adaptation is a process where the VM's
      operating system undergoes changes and adaptations to fully function on
      Compute Engine.
  """
    idleDuration = _messages.StringField(1)
    skipOsAdaptation = _messages.BooleanField(2)