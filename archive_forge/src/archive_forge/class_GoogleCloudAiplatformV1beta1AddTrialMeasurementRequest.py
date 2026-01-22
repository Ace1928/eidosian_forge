from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1AddTrialMeasurementRequest(_messages.Message):
    """Request message for VizierService.AddTrialMeasurement.

  Fields:
    measurement: Required. The measurement to be added to a Trial.
  """
    measurement = _messages.MessageField('GoogleCloudAiplatformV1beta1Measurement', 1)