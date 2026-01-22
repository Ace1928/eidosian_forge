from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.transfer import creds_util
from googlecloudsdk.command_lib.transfer import jobs_flag_util
from googlecloudsdk.command_lib.transfer import name_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
def _create_or_modify_schedule(job, args, messages, is_update, has_event_stream_flag):
    """Creates or modifies transfer Schedule object based on args."""
    schedule_starts = getattr(args, 'schedule_starts', None)
    schedule_repeats_every = getattr(args, 'schedule_repeats_every', None)
    schedule_repeats_until = getattr(args, 'schedule_repeats_until', None)
    has_schedule_flag = schedule_starts or schedule_repeats_every or schedule_repeats_until
    if has_schedule_flag:
        if not is_update and args.do_not_run:
            raise ValueError('Cannot set schedule and do-not-run flag.')
        if has_event_stream_flag:
            raise ValueError('Cannot set schedule and event stream.')
    if not is_update and args.do_not_run or has_event_stream_flag or (is_update and (not has_schedule_flag)):
        return
    if not job.schedule:
        job.schedule = messages.Schedule()
    if schedule_starts:
        start = schedule_starts.astimezone(times.UTC)
        job.schedule.scheduleStartDate = messages.Date(day=start.day, month=start.month, year=start.year)
        job.schedule.startTimeOfDay = messages.TimeOfDay(hours=start.hour, minutes=start.minute, seconds=start.second)
    elif not is_update:
        today_date = datetime.date.today()
        job.schedule.scheduleStartDate = messages.Date(day=today_date.day, month=today_date.month, year=today_date.year)
    if schedule_repeats_every:
        job.schedule.repeatInterval = '{}s'.format(schedule_repeats_every)
    if schedule_repeats_until:
        if not job.schedule.repeatInterval:
            raise ValueError('Scheduling a job end time requires setting a frequency with --schedule-repeats-every. If no job end time is set, the job will run one time.')
        end = schedule_repeats_until.astimezone(times.UTC)
        job.schedule.scheduleEndDate = messages.Date(day=end.day, month=end.month, year=end.year)
        job.schedule.endTimeOfDay = messages.TimeOfDay(hours=end.hour, minutes=end.minute, seconds=end.second)
    elif not is_update and (not job.schedule.repeatInterval):
        job.schedule.scheduleEndDate = job.schedule.scheduleStartDate