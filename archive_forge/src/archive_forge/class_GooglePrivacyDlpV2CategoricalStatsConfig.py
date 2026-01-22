from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CategoricalStatsConfig(_messages.Message):
    """Compute numerical stats over an individual column, including number of
  distinct values and value count distribution.

  Fields:
    field: Field to compute categorical stats on. All column types are
      supported except for arrays and structs. However, it may be more
      informative to use NumericalStats when the field type is supported,
      depending on the data.
  """
    field = _messages.MessageField('GooglePrivacyDlpV2FieldId', 1)