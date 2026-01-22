from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RddOperationGraph(_messages.Message):
    """Graph representing RDD dependencies. Consists of edges and a root
  cluster.

  Fields:
    edges: A RddOperationEdge attribute.
    incomingEdges: A RddOperationEdge attribute.
    outgoingEdges: A RddOperationEdge attribute.
    rootCluster: A RddOperationCluster attribute.
    stageId: A string attribute.
  """
    edges = _messages.MessageField('RddOperationEdge', 1, repeated=True)
    incomingEdges = _messages.MessageField('RddOperationEdge', 2, repeated=True)
    outgoingEdges = _messages.MessageField('RddOperationEdge', 3, repeated=True)
    rootCluster = _messages.MessageField('RddOperationCluster', 4)
    stageId = _messages.IntegerField(5)