from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceMeshAnalysisMessageBase(_messages.Message):
    """AnalysisMessageBase describes some common information that is needed for
  all messages.

  Enums:
    LevelValueValuesEnum: Represents how severe a message is.

  Fields:
    documentationUrl: A url pointing to the Service Mesh or Istio
      documentation for this specific error type.
    level: Represents how severe a message is.
    type: Represents the specific type of a message.
  """

    class LevelValueValuesEnum(_messages.Enum):
        """Represents how severe a message is.

    Values:
      LEVEL_UNSPECIFIED: Illegal. Same
        istio.analysis.v1alpha1.AnalysisMessageBase.Level.UNKNOWN.
      ERROR: ERROR represents a misconfiguration that must be fixed.
      WARNING: WARNING represents a misconfiguration that should be fixed.
      INFO: INFO represents an informational finding.
    """
        LEVEL_UNSPECIFIED = 0
        ERROR = 1
        WARNING = 2
        INFO = 3
    documentationUrl = _messages.StringField(1)
    level = _messages.EnumField('LevelValueValuesEnum', 2)
    type = _messages.MessageField('ServiceMeshType', 3)