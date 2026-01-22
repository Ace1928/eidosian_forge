from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestPlatformValueValuesEnum(_messages.Enum):
    """The platform of the test history. - In response: always set. Returns
    the platform of the last execution if unknown.

    Values:
      unknownPlatform: <no description>
      android: <no description>
      ios: <no description>
    """
    unknownPlatform = 0
    android = 1
    ios = 2