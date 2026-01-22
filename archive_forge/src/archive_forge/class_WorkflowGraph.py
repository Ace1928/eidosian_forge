from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowGraph(_messages.Message):
    """The workflow graph.

  Fields:
    nodes: Output only. The workflow nodes.
  """
    nodes = _messages.MessageField('WorkflowNode', 1, repeated=True)