from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PoolData(_messages.Message):
    """Pool Data

  Fields:
    name: A string attribute.
    stageIds: A string attribute.
  """
    name = _messages.StringField(1)
    stageIds = _messages.IntegerField(2, repeated=True)