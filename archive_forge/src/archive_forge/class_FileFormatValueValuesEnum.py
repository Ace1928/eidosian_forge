from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileFormatValueValuesEnum(_messages.Enum):
    """Output only. The requested file format.

    Values:
      IMPORT_RULES_FILE_FORMAT_UNSPECIFIED: Unspecified rules format.
      IMPORT_RULES_FILE_FORMAT_HARBOUR_BRIDGE_SESSION_FILE: HarbourBridge
        session file.
      IMPORT_RULES_FILE_FORMAT_ORATOPG_CONFIG_FILE: Ora2Pg configuration file.
    """
    IMPORT_RULES_FILE_FORMAT_UNSPECIFIED = 0
    IMPORT_RULES_FILE_FORMAT_HARBOUR_BRIDGE_SESSION_FILE = 1
    IMPORT_RULES_FILE_FORMAT_ORATOPG_CONFIG_FILE = 2