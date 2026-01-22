from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReportedParallelism(_messages.Message):
    """Represents the level of parallelism in a WorkItem's input, reported by
  the worker.

  Fields:
    isInfinite: Specifies whether the parallelism is infinite. If true,
      "value" is ignored. Infinite parallelism means the service will assume
      that the work item can always be split into more non-empty work items by
      dynamic splitting. This is a work-around for lack of support for
      infinity by the current JSON-based Java RPC stack.
    value: Specifies the level of parallelism in case it is finite.
  """
    isInfinite = _messages.BooleanField(1)
    value = _messages.FloatField(2)