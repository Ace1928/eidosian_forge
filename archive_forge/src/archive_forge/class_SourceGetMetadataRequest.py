from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceGetMetadataRequest(_messages.Message):
    """A request to compute the SourceMetadata of a Source.

  Fields:
    source: Specification of the source whose metadata should be computed.
  """
    source = _messages.MessageField('Source', 1)