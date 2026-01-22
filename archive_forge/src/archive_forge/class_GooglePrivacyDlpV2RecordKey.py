from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RecordKey(_messages.Message):
    """Message for a unique key indicating a record that contains a finding.

  Fields:
    bigQueryKey: Datastore key
    datastoreKey: BigQuery key
    idValues: Values of identifying columns in the given row. Order of values
      matches the order of `identifying_fields` specified in the scanning
      request.
  """
    bigQueryKey = _messages.MessageField('GooglePrivacyDlpV2BigQueryKey', 1)
    datastoreKey = _messages.MessageField('GooglePrivacyDlpV2DatastoreKey', 2)
    idValues = _messages.StringField(3, repeated=True)