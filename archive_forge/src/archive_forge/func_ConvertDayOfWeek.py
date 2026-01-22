from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as ex
def ConvertDayOfWeek(day):
    """Converts the user-given day-of-week into DayValueValuesEnum.

  Args:
    day: day of Week for weekly backup schdeule.

  Returns:
    DayValueValuesEnum.

  Raises:
    ValueError: if it is an invalid input.
  """
    day_num = arg_parsers.DayOfWeek.DAYS.index(day)
    messages = api_utils.GetMessages().GoogleFirestoreAdminV1WeeklyRecurrence()
    if day_num == 0:
        day_num = 7
    return messages.DayValueValuesEnum(day_num)