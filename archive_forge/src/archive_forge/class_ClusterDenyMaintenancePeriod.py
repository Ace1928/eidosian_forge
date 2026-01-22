from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterDenyMaintenancePeriod(_messages.Message):
    """ClusterDenyMaintenancePeriod definition. Except emergencies, maintenance
  will not be scheduled to start within this deny period. The start_date must
  be less than the end_date.

  Fields:
    endDate: Deny period end date. This can be: * A full date, with non-zero
      year, month and day values. * A month and day value, with a zero year
      for recurring. Date matching this period will have to be before the end.
    startDate: Deny period start date. This can be: * A full date, with non-
      zero year, month and day values. * A month and day value, with a zero
      year for recurring. Date matching this period will have to be the same
      or after the start.
    time: Time in UTC when the deny period starts on start_date and ends on
      end_date. This can be: * Full time.
  """
    endDate = _messages.MessageField('Date', 1)
    startDate = _messages.MessageField('Date', 2)
    time = _messages.MessageField('TimeOfDay', 3)