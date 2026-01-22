from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesBrokersGetRequest(_messages.Message):
    """A AnthoseventsNamespacesBrokersGetRequest object.

  Fields:
    name: The name of the Broker being retrieved.
  """
    name = _messages.StringField(1, required=True)