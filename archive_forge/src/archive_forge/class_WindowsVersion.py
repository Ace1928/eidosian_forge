from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WindowsVersion(_messages.Message):
    """Windows server version.

  Fields:
    imageType: Windows server image type
    osVersion: Windows server build number
    supportEndDate: Mainstream support end date
  """
    imageType = _messages.StringField(1)
    osVersion = _messages.StringField(2)
    supportEndDate = _messages.MessageField('Date', 3)