from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3FulfillmentSetParameterAction(_messages.Message):
    """Setting a parameter value.

  Fields:
    parameter: Display name of the parameter.
    value: The new value of the parameter. A null value clears the parameter.
  """
    parameter = _messages.StringField(1)
    value = _messages.MessageField('extra_types.JsonValue', 2)