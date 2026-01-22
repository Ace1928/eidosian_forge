from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesReadBlobDataRequest(_messages.Message):
    """A AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesReadBl
  obDataRequest object.

  Fields:
    blobIds: IDs of the blobs to read.
    timeSeries: Required. The resource name of the TensorboardTimeSeries to
      list Blobs. Format: `projects/{project}/locations/{location}/tensorboard
      s/{tensorboard}/experiments/{experiment}/runs/{run}/timeSeries/{time_ser
      ies}`
  """
    blobIds = _messages.StringField(1, repeated=True)
    timeSeries = _messages.StringField(2, required=True)