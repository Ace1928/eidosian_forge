from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsKuberunsPatchRequest(_messages.Message):
    """A AnthoseventsKuberunsPatchRequest object.

  Fields:
    kubeRun: A KubeRun resource to be passed as the request body.
    name: The name of the KubeRun resource being updated.
  """
    kubeRun = _messages.MessageField('KubeRun', 1)
    name = _messages.StringField(2, required=True)