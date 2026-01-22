from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudStorageFilters(_messages.Message):
    """Options to filter data on storage systems. Next ID: 2

  Fields:
    bucket: Bucket for which the report will be generated.
  """
    bucket = _messages.StringField(1)