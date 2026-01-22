from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InfoType(_messages.Message):
    """Type of information detected by the API.

  Fields:
    name: Name of the information type. Either a name of your choosing when
      creating a CustomInfoType, or one of the names listed at
      https://cloud.google.com/sensitive-data-protection/docs/infotypes-
      reference when specifying a built-in type. When sending Cloud DLP
      results to Data Catalog, infoType names should conform to the pattern
      `[A-Za-z0-9$_-]{1,64}`.
    sensitivityScore: Optional custom sensitivity for this InfoType. This only
      applies to data profiling.
    version: Optional version name for this InfoType.
  """
    name = _messages.StringField(1)
    sensitivityScore = _messages.MessageField('GooglePrivacyDlpV2SensitivityScore', 2)
    version = _messages.StringField(3)