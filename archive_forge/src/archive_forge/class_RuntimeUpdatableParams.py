from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeUpdatableParams(_messages.Message):
    """Additional job parameters that can only be updated during runtime using
  the projects.jobs.update method. These fields have no effect when specified
  during job creation.

  Fields:
    maxNumWorkers: The maximum number of workers to cap autoscaling at. This
      field is currently only supported for Streaming Engine jobs.
    minNumWorkers: The minimum number of workers to scale down to. This field
      is currently only supported for Streaming Engine jobs.
    workerUtilizationHint: Target worker utilization, compared against the
      aggregate utilization of the worker pool by autoscaler, to determine
      upscaling and downscaling when absent other constraints such as backlog.
      For more information, see [Update an existing
      pipeline](https://cloud.google.com/dataflow/docs/guides/updating-a-
      pipeline).
  """
    maxNumWorkers = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minNumWorkers = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    workerUtilizationHint = _messages.FloatField(3)