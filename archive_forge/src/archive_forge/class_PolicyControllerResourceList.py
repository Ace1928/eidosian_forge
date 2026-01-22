from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyControllerResourceList(_messages.Message):
    """ResourceList contains container resource requirements.

  Fields:
    cpu: CPU requirement expressed in Kubernetes resource units.
    memory: Memory requirement expressed in Kubernetes resource units.
  """
    cpu = _messages.StringField(1)
    memory = _messages.StringField(2)