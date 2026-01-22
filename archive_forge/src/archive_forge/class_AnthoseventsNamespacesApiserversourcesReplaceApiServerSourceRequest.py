from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesApiserversourcesReplaceApiServerSourceRequest(_messages.Message):
    """A AnthoseventsNamespacesApiserversourcesReplaceApiServerSourceRequest
  object.

  Fields:
    apiServerSource: A ApiServerSource resource to be passed as the request
      body.
    name: The name of the apiserversource being retrieved. If needed, replace
      {namespace_id} with the project ID.
    region: The region in which this resource exists.
  """
    apiServerSource = _messages.MessageField('ApiServerSource', 1)
    name = _messages.StringField(2, required=True)
    region = _messages.StringField(3)