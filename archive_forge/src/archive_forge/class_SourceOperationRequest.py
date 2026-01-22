from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceOperationRequest(_messages.Message):
    """A work item that represents the different operations that can be
  performed on a user-defined Source specification.

  Fields:
    getMetadata: Information about a request to get metadata about a source.
    name: User-provided name of the Read instruction for this source.
    originalName: System-defined name for the Read instruction for this source
      in the original workflow graph.
    split: Information about a request to split a source.
    stageName: System-defined name of the stage containing the source
      operation. Unique across the workflow.
    systemName: System-defined name of the Read instruction for this source.
      Unique across the workflow.
  """
    getMetadata = _messages.MessageField('SourceGetMetadataRequest', 1)
    name = _messages.StringField(2)
    originalName = _messages.StringField(3)
    split = _messages.MessageField('SourceSplitRequest', 4)
    stageName = _messages.StringField(5)
    systemName = _messages.StringField(6)