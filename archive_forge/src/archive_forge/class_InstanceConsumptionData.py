from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceConsumptionData(_messages.Message):
    """A InstanceConsumptionData object.

  Fields:
    consumptionInfo: Resources consumed by the instance.
    instance: Server-defined URL for the instance.
  """
    consumptionInfo = _messages.MessageField('InstanceConsumptionInfo', 1)
    instance = _messages.StringField(2)