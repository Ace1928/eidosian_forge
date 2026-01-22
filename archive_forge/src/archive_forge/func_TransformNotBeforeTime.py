from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from dateutil import tz
from googlecloudsdk.core.util import times
def TransformNotBeforeTime(subject_description):
    """Use this function in a display transform to truncate anything smaller than minutes from ISO8601 timestamp."""
    if subject_description and 'notBeforeTime' in subject_description:
        return times.ParseDateTime(subject_description.get('notBeforeTime')).astimezone(tz.tzutc()).strftime('%Y-%m-%dT%H:%MZ')
    return ''