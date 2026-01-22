from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReservationsResizeRequest(_messages.Message):
    """A ReservationsResizeRequest object.

  Fields:
    specificSkuCount: Number of allocated resources can be resized with
      minimum = 1 and maximum = 1000.
  """
    specificSkuCount = _messages.IntegerField(1)