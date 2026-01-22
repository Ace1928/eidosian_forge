from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CitationMetadata(_messages.Message):
    """The schema of citations found in textual prediction outputs. Citations
  originate from various sources and indicate that these contents should be
  cited properly.

  Fields:
    citations: Metadata of all citations found in this prediction output.
  """
    citations = _messages.MessageField('Citation', 1, repeated=True)