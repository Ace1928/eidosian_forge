from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateOperatorProgress(_messages.Message):
    """A StateOperatorProgress object.

  Messages:
    CustomMetricsValue: A CustomMetricsValue object.

  Fields:
    allRemovalsTimeMs: A string attribute.
    allUpdatesTimeMs: A string attribute.
    commitTimeMs: A string attribute.
    customMetrics: A CustomMetricsValue attribute.
    memoryUsedBytes: A string attribute.
    numRowsDroppedByWatermark: A string attribute.
    numRowsRemoved: A string attribute.
    numRowsTotal: A string attribute.
    numRowsUpdated: A string attribute.
    numShufflePartitions: A string attribute.
    numStateStoreInstances: A string attribute.
    operatorName: A string attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CustomMetricsValue(_messages.Message):
        """A CustomMetricsValue object.

    Messages:
      AdditionalProperty: An additional property for a CustomMetricsValue
        object.

    Fields:
      additionalProperties: Additional properties of type CustomMetricsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CustomMetricsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.IntegerField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    allRemovalsTimeMs = _messages.IntegerField(1)
    allUpdatesTimeMs = _messages.IntegerField(2)
    commitTimeMs = _messages.IntegerField(3)
    customMetrics = _messages.MessageField('CustomMetricsValue', 4)
    memoryUsedBytes = _messages.IntegerField(5)
    numRowsDroppedByWatermark = _messages.IntegerField(6)
    numRowsRemoved = _messages.IntegerField(7)
    numRowsTotal = _messages.IntegerField(8)
    numRowsUpdated = _messages.IntegerField(9)
    numShufflePartitions = _messages.IntegerField(10)
    numStateStoreInstances = _messages.IntegerField(11)
    operatorName = _messages.StringField(12)