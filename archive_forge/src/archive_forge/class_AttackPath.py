from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttackPath(_messages.Message):
    """A path that an attacker could take to reach an exposed resource.

  Fields:
    edges: A list of the edges between nodes in this attack path.
    name: The attack path name, for example,
      `organizations/12/simulations/34/valuedResources/56/attackPaths/78`
    pathNodes: A list of nodes that exist in this attack path.
  """
    edges = _messages.MessageField('AttackPathEdge', 1, repeated=True)
    name = _messages.StringField(2)
    pathNodes = _messages.MessageField('AttackPathNode', 3, repeated=True)