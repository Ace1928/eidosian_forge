from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExportPreviewResultResponse(_messages.Message):
    """A response to `ExportPreviewResult` call. Contains preview results.

  Fields:
    result: Output only. Signed URLs for accessing the plan files.
  """
    result = _messages.MessageField('PreviewResult', 1)