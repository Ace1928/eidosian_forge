from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute.os_config import utils as osconfig_api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.os_config import utils as osconfig_command_utils
from googlecloudsdk.core import properties
def _TransformNextRun(resource):
    """Returns the timestamp of the next scheduled run for this patch deployment."""
    if _ONE_TIME_SCHEDULE in resource:
        if resource.get(_LAST_EXECUTE_TIME, ''):
            return _DEFAULT_NO_VALUE
        else:
            schedule = resource[_ONE_TIME_SCHEDULE]
            return schedule.get('executeTime', _DEFAULT_NO_VALUE)
    elif _RECURRING_SCHEDULE in resource:
        schedule = resource[_RECURRING_SCHEDULE]
        return schedule.get(_NEXT_EXECUTE_TIME, _DEFAULT_NO_VALUE)
    else:
        return _DEFAULT_NO_VALUE