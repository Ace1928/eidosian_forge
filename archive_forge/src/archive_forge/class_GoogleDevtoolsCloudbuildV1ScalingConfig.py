from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1ScalingConfig(_messages.Message):
    """Defines the scaling configuration for the pool.

  Fields:
    maxWorkersPerZone: Max number of workers in the Private Pool per zone.
      Cloud Build will run workloads in three zones per Private Pool for
      reliability.
    readyWorkers: The number of preemptible workers (pods) that will run with
      the minimum vCPU and memory to keep resources ready for customer
      workloads in the cluster. If unset, a value of 0 will be used.
  """
    maxWorkersPerZone = _messages.IntegerField(1)
    readyWorkers = _messages.IntegerField(2)