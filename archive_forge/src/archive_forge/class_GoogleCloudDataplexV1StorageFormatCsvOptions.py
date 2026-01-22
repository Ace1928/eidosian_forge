from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1StorageFormatCsvOptions(_messages.Message):
    """Describes CSV and similar semi-structured data formats.

  Fields:
    delimiter: Optional. The delimiter used to separate values. Defaults to
      ','.
    encoding: Optional. The character encoding of the data. Accepts "US-
      ASCII", "UTF-8", and "ISO-8859-1". Defaults to UTF-8 if unspecified.
    headerRows: Optional. The number of rows to interpret as header rows that
      should be skipped when reading data rows. Defaults to 0.
    quote: Optional. The character used to quote column values. Accepts '"'
      (double quotation mark) or ''' (single quotation mark). Defaults to '"'
      (double quotation mark) if unspecified.
  """
    delimiter = _messages.StringField(1)
    encoding = _messages.StringField(2)
    headerRows = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    quote = _messages.StringField(4)