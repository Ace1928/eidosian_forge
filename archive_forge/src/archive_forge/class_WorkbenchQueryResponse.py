from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkbenchQueryResponse(_messages.Message):
    """Response to querying a Workbench

  Fields:
    candidates: Output only. Candidate responses from the model.
    citationMetadata: Output only. Citation metadata. Contains citation
      information of `content`.
    response: Output only. Response to the user's query.
  """
    candidates = _messages.MessageField('Candidate', 1, repeated=True)
    citationMetadata = _messages.MessageField('CitationMetadata', 2)
    response = _messages.StringField(3)