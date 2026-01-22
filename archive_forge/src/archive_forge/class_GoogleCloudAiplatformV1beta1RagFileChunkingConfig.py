from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RagFileChunkingConfig(_messages.Message):
    """Specifies the size and overlap of chunks for RagFiles.

  Fields:
    chunkOverlap: The overlap between chunks.
    chunkSize: The size of the chunks.
  """
    chunkOverlap = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    chunkSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)