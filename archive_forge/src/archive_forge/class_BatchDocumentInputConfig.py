from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchDocumentInputConfig(_messages.Message):
    """Input configuration for BatchTranslateDocument request.

  Fields:
    gcsSource: Google Cloud Storage location for the source input. This can be
      a single file (for example, `gs://translation-test/input.docx`) or a
      wildcard (for example, `gs://translation-test/*`). File mime type is
      determined based on extension. Supported mime type includes: - `pdf`,
      application/pdf - `docx`, application/vnd.openxmlformats-
      officedocument.wordprocessingml.document - `pptx`,
      application/vnd.openxmlformats-
      officedocument.presentationml.presentation - `xlsx`,
      application/vnd.openxmlformats-officedocument.spreadsheetml.sheet The
      max file size to support for `.docx`, `.pptx` and `.xlsx` is 100MB. The
      max file size to support for `.pdf` is 1GB and the max page limit is
      1000 pages. The max file size to support for all input documents is 1GB.
  """
    gcsSource = _messages.MessageField('GcsSource', 1)