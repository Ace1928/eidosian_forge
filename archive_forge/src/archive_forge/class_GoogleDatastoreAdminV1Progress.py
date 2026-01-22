from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1Progress(_messages.Message):
    """Measures the progress of a particular metric.

  Fields:
    workCompleted: The amount of work that has been completed. Note that this
      may be greater than work_estimated.
    workEstimated: An estimate of how much work needs to be performed. May be
      zero if the work estimate is unavailable.
  """
    workCompleted = _messages.IntegerField(1)
    workEstimated = _messages.IntegerField(2)