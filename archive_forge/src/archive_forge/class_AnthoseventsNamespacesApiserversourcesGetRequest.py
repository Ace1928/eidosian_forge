from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesApiserversourcesGetRequest(_messages.Message):
    """A AnthoseventsNamespacesApiserversourcesGetRequest object.

  Fields:
    name: The name of the apiserversource being retrieved. If needed, replace
      {namespace_id} with the project ID.
    region: The region in which this resource exists.
  """
    name = _messages.StringField(1, required=True)
    region = _messages.StringField(2)