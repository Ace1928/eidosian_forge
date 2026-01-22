from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceProfileInfo(_messages.Message):
    """Resource profile that contains information about all the resources
  required by executors and tasks.

  Messages:
    ExecutorResourcesValue: A ExecutorResourcesValue object.
    TaskResourcesValue: A TaskResourcesValue object.

  Fields:
    executorResources: A ExecutorResourcesValue attribute.
    resourceProfileId: A integer attribute.
    taskResources: A TaskResourcesValue attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExecutorResourcesValue(_messages.Message):
        """A ExecutorResourcesValue object.

    Messages:
      AdditionalProperty: An additional property for a ExecutorResourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ExecutorResourcesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExecutorResourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A ExecutorResourceRequest attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ExecutorResourceRequest', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TaskResourcesValue(_messages.Message):
        """A TaskResourcesValue object.

    Messages:
      AdditionalProperty: An additional property for a TaskResourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type TaskResourcesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TaskResourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A TaskResourceRequest attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('TaskResourceRequest', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    executorResources = _messages.MessageField('ExecutorResourcesValue', 1)
    resourceProfileId = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    taskResources = _messages.MessageField('TaskResourcesValue', 3)