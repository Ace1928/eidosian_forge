from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsKuberunsGetRequest(_messages.Message):
    """A AnthoseventsKuberunsGetRequest object.

  Fields:
    name: The name of the KubeRun resource being retrieved.
  """
    name = _messages.StringField(1, required=True)