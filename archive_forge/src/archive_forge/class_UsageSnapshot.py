from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UsageSnapshot(_messages.Message):
    """The usage snapshot represents the resources consumed by a workload at a
  specified time.

  Fields:
    acceleratorType: Optional. Accelerator type being used, if any
    milliAccelerator: Optional. Milli (one-thousandth) accelerator. (see
      Dataproc Serverless pricing (https://cloud.google.com/dataproc-
      serverless/pricing))
    milliDcu: Optional. Milli (one-thousandth) Dataproc Compute Units (DCUs)
      (see Dataproc Serverless pricing (https://cloud.google.com/dataproc-
      serverless/pricing)).
    milliDcuPremium: Optional. Milli (one-thousandth) Dataproc Compute Units
      (DCUs) charged at premium tier (see Dataproc Serverless pricing
      (https://cloud.google.com/dataproc-serverless/pricing)).
    shuffleStorageGb: Optional. Shuffle Storage in gigabytes (GB). (see
      Dataproc Serverless pricing (https://cloud.google.com/dataproc-
      serverless/pricing))
    shuffleStorageGbPremium: Optional. Shuffle Storage in gigabytes (GB)
      charged at premium tier. (see Dataproc Serverless pricing
      (https://cloud.google.com/dataproc-serverless/pricing))
    snapshotTime: Optional. The timestamp of the usage snapshot.
  """
    acceleratorType = _messages.StringField(1)
    milliAccelerator = _messages.IntegerField(2)
    milliDcu = _messages.IntegerField(3)
    milliDcuPremium = _messages.IntegerField(4)
    shuffleStorageGb = _messages.IntegerField(5)
    shuffleStorageGbPremium = _messages.IntegerField(6)
    snapshotTime = _messages.StringField(7)