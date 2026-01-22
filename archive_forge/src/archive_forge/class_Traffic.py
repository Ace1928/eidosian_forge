from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Traffic(_messages.Message):
    """Expected traffic between two coordinates.

  Fields:
    peakTraffic: Expected peak traffic.
  """
    peakTraffic = _messages.MessageField('PeakTraffic', 1)