from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObjectMetadataReportOptions(_messages.Message):
    """Report specification for exporting object metadata. Next ID: 4

  Fields:
    metadataFields: Metadata fields to be included in the report.
    storageDestinationOptions: Cloud Storage as the storage system.
    storageFilters: Cloud Storage as the storage system.
  """
    metadataFields = _messages.StringField(1, repeated=True)
    storageDestinationOptions = _messages.MessageField('CloudStorageDestinationOptions', 2)
    storageFilters = _messages.MessageField('CloudStorageFilters', 3)