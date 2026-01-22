from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DimensionMetric(_messages.Message):
    """Encapsulates a metric grouped by dimension.

  Fields:
    individualNames: Individual dimension names. E.g. ["dim1_name",
      "dim2_name"].
    metrics: List of metrics.
    name: Comma joined dimension names. E.g. "dim1_name,dim2_name".
      Deprecated. If name already has comma before join, we may get wrong
      splits. Please use individual_names.
  """
    individualNames = _messages.StringField(1, repeated=True)
    metrics = _messages.MessageField('GoogleCloudApigeeV1Metric', 2, repeated=True)
    name = _messages.StringField(3)