from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2StatisticalTable(_messages.Message):
    """An auxiliary table containing statistical information on the relative
  frequency of different quasi-identifiers values. It has one or several
  quasi-identifiers columns, and one column that indicates the relative
  frequency of each quasi-identifier tuple. If a tuple is present in the data
  but not in the auxiliary table, the corresponding relative frequency is
  assumed to be zero (and thus, the tuple is highly reidentifiable).

  Fields:
    quasiIds: Required. Quasi-identifier columns.
    relativeFrequency: Required. The relative frequency column must contain a
      floating-point number between 0 and 1 (inclusive). Null values are
      assumed to be zero.
    table: Required. Auxiliary table location.
  """
    quasiIds = _messages.MessageField('GooglePrivacyDlpV2QuasiIdentifierField', 1, repeated=True)
    relativeFrequency = _messages.MessageField('GooglePrivacyDlpV2FieldId', 2)
    table = _messages.MessageField('GooglePrivacyDlpV2BigQueryTable', 3)