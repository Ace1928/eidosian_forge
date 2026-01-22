from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceUtilizationReport(_messages.Message):
    """Worker metrics exported from workers. This contains resource utilization
  metrics accumulated from a variety of sources. For more information, see
  go/df-resource-signals.

  Messages:
    ContainersValue: Per container information. Key: container name.

  Fields:
    containers: Per container information. Key: container name.
    cpuTime: CPU utilization samples.
    memoryInfo: Memory utilization samples.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ContainersValue(_messages.Message):
        """Per container information. Key: container name.

    Messages:
      AdditionalProperty: An additional property for a ContainersValue object.

    Fields:
      additionalProperties: Additional properties of type ContainersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ContainersValue object.

      Fields:
        key: Name of the additional property.
        value: A ResourceUtilizationReport attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ResourceUtilizationReport', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    containers = _messages.MessageField('ContainersValue', 1)
    cpuTime = _messages.MessageField('CPUTime', 2, repeated=True)
    memoryInfo = _messages.MessageField('MemInfo', 3, repeated=True)