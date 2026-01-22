from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleSheetsOptions(_messages.Message):
    """A GoogleSheetsOptions object.

  Fields:
    skipLeadingRows: [Optional] The number of rows at the top of a sheet that
      BigQuery will skip when reading the data. The default value is 0. This
      property is useful if you have header rows that should be skipped. When
      autodetect is on, behavior is the following: * skipLeadingRows
      unspecified - Autodetect tries to detect headers in the first row. If
      they are not detected, the row is read as data. Otherwise data is read
      starting from the second row. * skipLeadingRows is 0 - Instructs
      autodetect that there are no headers and data should be read starting
      from the first row. * skipLeadingRows = N > 0 - Autodetect skips N-1
      rows and tries to detect headers in row N. If headers are not detected,
      row N is just skipped. Otherwise row N is used to extract column names
      for the detected schema.
  """
    skipLeadingRows = _messages.IntegerField(1)