from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MapTask(_messages.Message):
    """MapTask consists of an ordered set of instructions, each of which
  describes one particular low-level operation for the worker to perform in
  order to accomplish the MapTask's WorkItem. Each instruction must appear in
  the list before any instructions which depends on its output.

  Fields:
    counterPrefix: Counter prefix that can be used to prefix counters. Not
      currently used in Dataflow.
    instructions: The instructions in the MapTask.
    stageName: System-defined name of the stage containing this MapTask.
      Unique across the workflow.
    systemName: System-defined name of this MapTask. Unique across the
      workflow.
  """
    counterPrefix = _messages.StringField(1)
    instructions = _messages.MessageField('ParallelInstruction', 2, repeated=True)
    stageName = _messages.StringField(3)
    systemName = _messages.StringField(4)