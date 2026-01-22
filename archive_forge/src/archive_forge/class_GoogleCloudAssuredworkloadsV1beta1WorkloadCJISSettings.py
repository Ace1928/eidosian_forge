from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1beta1WorkloadCJISSettings(_messages.Message):
    """Settings specific to resources needed for CJIS.

  Fields:
    kmsSettings: Input only. Immutable. Settings used to create a CMEK crypto
      key.
  """
    kmsSettings = _messages.MessageField('GoogleCloudAssuredworkloadsV1beta1WorkloadKMSSettings', 1)