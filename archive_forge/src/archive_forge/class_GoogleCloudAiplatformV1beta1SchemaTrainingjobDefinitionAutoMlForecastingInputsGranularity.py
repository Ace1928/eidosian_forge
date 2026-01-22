from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputsGranularity(_messages.Message):
    """A duration of time expressed in time granularity units.

  Fields:
    quantity: The number of granularity_units between data points in the
      training data. If `granularity_unit` is `minute`, can be 1, 5, 10, 15,
      or 30. For all other values of `granularity_unit`, must be 1.
    unit: The time granularity unit of this time period. The supported units
      are: * "minute" * "hour" * "day" * "week" * "month" * "year"
  """
    quantity = _messages.IntegerField(1)
    unit = _messages.StringField(2)