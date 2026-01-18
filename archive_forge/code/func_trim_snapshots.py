from __future__ import print_function
from boto.sdb.db.model import Model
from boto.sdb.db.property import StringProperty, IntegerProperty, ListProperty, ReferenceProperty, CalculatedProperty
from boto.manage.server import Server
from boto.manage import propget
import boto.utils
import boto.ec2
import time
import traceback
from contextlib import closing
import datetime
def trim_snapshots(self, delete=False):
    """
        Trim the number of snapshots for this volume.  This method always
        keeps the oldest snapshot.  It then uses the parameters passed in
        to determine how many others should be kept.

        The algorithm is to keep all snapshots from the current day.  Then
        it will keep the first snapshot of the day for the previous seven days.
        Then, it will keep the first snapshot of the week for the previous
        four weeks.  After than, it will keep the first snapshot of the month
        for as many months as there are.

        """
    snaps = self.get_snapshots()
    if len(snaps) <= 2:
        return snaps
    snaps = snaps[1:-1]
    now = datetime.datetime.now(snaps[0].date.tzinfo)
    midnight = datetime.datetime(year=now.year, month=now.month, day=now.day, tzinfo=now.tzinfo)
    one_week = datetime.timedelta(days=7, seconds=60 * 60)
    print(midnight - one_week, midnight)
    previous_week = self.get_snapshot_range(snaps, midnight - one_week, midnight)
    print(previous_week)
    if not previous_week:
        return snaps
    current_day = None
    for snap in previous_week:
        if current_day and current_day == snap.date.day:
            snap.keep = False
        else:
            current_day = snap.date.day
    if previous_week:
        week_boundary = previous_week[0].date
        if week_boundary.weekday() != 0:
            delta = datetime.timedelta(days=week_boundary.weekday())
            week_boundary = week_boundary - delta
    partial_week = self.get_snapshot_range(snaps, week_boundary, previous_week[0].date)
    if len(partial_week) > 1:
        for snap in partial_week[1:]:
            snap.keep = False
    for i in range(0, 4):
        weeks_worth = self.get_snapshot_range(snaps, week_boundary - one_week, week_boundary)
        if len(weeks_worth) > 1:
            for snap in weeks_worth[1:]:
                snap.keep = False
        week_boundary = week_boundary - one_week
    remainder = self.get_snapshot_range(snaps, end_date=week_boundary)
    current_month = None
    for snap in remainder:
        if current_month and current_month == snap.date.month:
            snap.keep = False
        else:
            current_month = snap.date.month
    if delete:
        for snap in snaps:
            if not snap.keep:
                boto.log.info('Deleting %s(%s) for %s' % (snap, snap.date, self.name))
                snap.delete()
    return snaps