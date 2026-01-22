from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ByteContentItem(_messages.Message):
    """Container for bytes to inspect or redact.

  Enums:
    TypeValueValuesEnum: The type of data stored in the bytes string. Default
      will be TEXT_UTF8.

  Fields:
    data: Content data to inspect or redact.
    type: The type of data stored in the bytes string. Default will be
      TEXT_UTF8.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of data stored in the bytes string. Default will be
    TEXT_UTF8.

    Values:
      BYTES_TYPE_UNSPECIFIED: Unused
      IMAGE: Any image type.
      IMAGE_JPEG: jpeg
      IMAGE_BMP: bmp
      IMAGE_PNG: png
      IMAGE_SVG: svg
      TEXT_UTF8: plain text
      WORD_DOCUMENT: docx, docm, dotx, dotm
      PDF: pdf
      POWERPOINT_DOCUMENT: pptx, pptm, potx, potm, pot
      EXCEL_DOCUMENT: xlsx, xlsm, xltx, xltm
      AVRO: avro
      CSV: csv
      TSV: tsv
    """
        BYTES_TYPE_UNSPECIFIED = 0
        IMAGE = 1
        IMAGE_JPEG = 2
        IMAGE_BMP = 3
        IMAGE_PNG = 4
        IMAGE_SVG = 5
        TEXT_UTF8 = 6
        WORD_DOCUMENT = 7
        PDF = 8
        POWERPOINT_DOCUMENT = 9
        EXCEL_DOCUMENT = 10
        AVRO = 11
        CSV = 12
        TSV = 13
    data = _messages.BytesField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)