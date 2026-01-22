from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FutureReservationSpecificSKUProperties(_messages.Message):
    """A FutureReservationSpecificSKUProperties object.

  Fields:
    instanceProperties: Properties of the SKU instances being reserved.
    sourceInstanceTemplate: The instance template that will be used to
      populate the ReservedInstanceProperties of the future reservation
    totalCount: Total number of instances for which capacity assurance is
      requested at a future time period.
  """
    instanceProperties = _messages.MessageField('AllocationSpecificSKUAllocationReservedInstanceProperties', 1)
    sourceInstanceTemplate = _messages.StringField(2)
    totalCount = _messages.IntegerField(3)