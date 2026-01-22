from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TasksValue(_messages.Message):
    """A TasksValue object.

    Messages:
      AdditionalProperty: An additional property for a TasksValue object.

    Fields:
      additionalProperties: Additional properties of type TasksValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TasksValue object.

      Fields:
        key: Name of the additional property.
        value: A TaskData attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('TaskData', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)