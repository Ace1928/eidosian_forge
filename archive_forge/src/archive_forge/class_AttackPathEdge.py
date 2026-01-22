from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttackPathEdge(_messages.Message):
    """Represents a connection between a source node and a destination node in
  this attack path.

  Fields:
    destination: The attack node uuid of the destination node.
    source: The attack node uuid of the source node.
  """
    destination = _messages.StringField(1)
    source = _messages.StringField(2)