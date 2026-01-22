from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionHierarchyConfig(_messages.Message):
    """Configuration that defines the hierarchical relationship of time series
  and parameters for hierarchical forecasting strategies.

  Fields:
    groupColumns: A list of time series attribute column names that define the
      time series hierarchy. Only one level of hierarchy is supported, ex.
      'region' for a hierarchy of stores or 'department' for a hierarchy of
      products. If multiple columns are specified, time series will be grouped
      by their combined values, ex. ('blue', 'large') for 'color' and 'size',
      up to 5 columns are accepted. If no group columns are specified, all
      time series are considered to be part of the same group.
    groupTemporalTotalWeight: The weight of the loss for predictions
      aggregated over both the horizon and time series in the same hierarchy
      group.
    groupTotalWeight: The weight of the loss for predictions aggregated over
      time series in the same group.
    temporalTotalWeight: The weight of the loss for predictions aggregated
      over the horizon for a single time series.
  """
    groupColumns = _messages.StringField(1, repeated=True)
    groupTemporalTotalWeight = _messages.FloatField(2)
    groupTotalWeight = _messages.FloatField(3)
    temporalTotalWeight = _messages.FloatField(4)