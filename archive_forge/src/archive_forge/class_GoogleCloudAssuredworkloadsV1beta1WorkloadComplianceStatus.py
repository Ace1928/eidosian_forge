from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1WorkloadComplianceStatus(_messages.Message):
    """Represents the Compliance Status of this workload

  Fields:
    acknowledgedResourceViolationCount: Number of current resource violations
      which are not acknowledged.
    acknowledgedViolationCount: Number of current orgPolicy violations which
      are acknowledged.
    activeResourceViolationCount: Number of current resource violations which
      are acknowledged.
    activeViolationCount: Number of current orgPolicy violations which are not
      acknowledged.
  """
    acknowledgedResourceViolationCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    acknowledgedViolationCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    activeResourceViolationCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    activeViolationCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)