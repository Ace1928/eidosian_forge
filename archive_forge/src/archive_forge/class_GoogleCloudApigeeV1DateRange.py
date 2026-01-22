from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1DateRange(_messages.Message):
    """Date range of the data to export.

  Fields:
    end: Required. End date (exclusive) of the data to export in the format
      `yyyy-mm-dd`. The date range ends at 00:00:00 UTC on the end date- which
      will not be in the output.
    start: Required. Start date of the data to export in the format `yyyy-mm-
      dd`. The date range begins at 00:00:00 UTC on the start date.
  """
    end = _messages.StringField(1)
    start = _messages.StringField(2)