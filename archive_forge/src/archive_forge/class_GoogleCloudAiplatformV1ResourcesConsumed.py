from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ResourcesConsumed(_messages.Message):
    """Statistics information about resource consumption.

  Fields:
    replicaHours: Output only. The number of replica hours used. Note that
      many replicas may run in parallel, and additionally any given work may
      be queued for some time. Therefore this value is not strictly related to
      wall time.
  """
    replicaHours = _messages.FloatField(1)