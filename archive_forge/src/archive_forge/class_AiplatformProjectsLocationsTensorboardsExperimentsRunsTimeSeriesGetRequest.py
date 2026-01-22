from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesGetRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesGetRequest
  object.

  Fields:
    name: Required. The name of the TensorboardTimeSeries resource. Format: `p
      rojects/{project}/locations/{location}/tensorboards/{tensorboard}/experi
      ments/{experiment}/runs/{run}/timeSeries/{time_series}`
  """
    name = _messages.StringField(1, required=True)