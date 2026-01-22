from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DenyMaintenancePeriod(_messages.Message):
    """DenyMaintenancePeriod definition. Maintenance is forbidden within the
  deny period. The start_date must be less than the end_date.

  Fields:
    endDate: Deny period end date. This can be: * A full date, with non-zero
      year, month and day values. * A month and day value, with a zero year.
      Allows recurring deny periods each year. Date matching this period will
      have to be before the end.
    startDate: Deny period start date. This can be: * A full date, with non-
      zero year, month and day values. * A month and day value, with a zero
      year. Allows recurring deny periods each year. Date matching this period
      will have to be the same or after the start.
    time: Time in UTC when the Blackout period starts on start_date and ends
      on end_date. This can be: * Full time. * All zeros for 00:00:00 UTC
  """
    endDate = _messages.MessageField('Date', 1)
    startDate = _messages.MessageField('Date', 2)
    time = _messages.MessageField('TimeOfDay', 3)