from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaAutoscale(_messages.Message):
    """Autoscale defines the autoscaling policy of a worker pool.

  Fields:
    maxSize: Optional. The maximal number of workers. Must be equal to or
      greater than min_size.
    minIdleWorkers: Optional. The minimum number of idle workers the
      autoscaler will aim to have in the pool at all times that are
      immediately available to accept a surge in build traffic. The pool size
      will still be constrained by min_size and max_size.
    minSize: Optional. The minimal number of workers. Must be greater than 0.
  """
    maxSize = _messages.IntegerField(1)
    minIdleWorkers = _messages.IntegerField(2)
    minSize = _messages.IntegerField(3)