from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsBatchReadRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsBatchReadRequest object.

  Fields:
    tensorboard: Required. The resource name of the Tensorboard containing
      TensorboardTimeSeries to read data from. Format:
      `projects/{project}/locations/{location}/tensorboards/{tensorboard}`.
      The TensorboardTimeSeries referenced by time_series must be sub
      resources of this Tensorboard.
    timeSeries: Required. The resource names of the TensorboardTimeSeries to
      read data from. Format: `projects/{project}/locations/{location}/tensorb
      oards/{tensorboard}/experiments/{experiment}/runs/{run}/timeSeries/{time
      _series}`
  """
    tensorboard = _messages.StringField(1, required=True)
    timeSeries = _messages.StringField(2, repeated=True)