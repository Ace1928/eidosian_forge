from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerfUnitValueValuesEnum(_messages.Enum):
    """PerfUnitValueValuesEnum enum type.

    Values:
      perfUnitUnspecified: <no description>
      kibibyte: <no description>
      percent: <no description>
      bytesPerSecond: <no description>
      framesPerSecond: <no description>
      byte: <no description>
    """
    perfUnitUnspecified = 0
    kibibyte = 1
    percent = 2
    bytesPerSecond = 3
    framesPerSecond = 4
    byte = 5