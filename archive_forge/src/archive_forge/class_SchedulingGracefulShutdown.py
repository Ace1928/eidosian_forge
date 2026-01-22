from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SchedulingGracefulShutdown(_messages.Message):
    """Configuration for gracefully shutting down the instance.

  Fields:
    enabled: Opts-in for graceful shutdown.
    maxDuration: Specifies time needed to gracefully shut down the instance.
      After that time, the instance goes to STOPPING even if graceful shutdown
      is not completed.
  """
    enabled = _messages.BooleanField(1)
    maxDuration = _messages.MessageField('Duration', 2)