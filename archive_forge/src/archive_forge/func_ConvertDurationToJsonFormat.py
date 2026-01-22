from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
import six
def ConvertDurationToJsonFormat(maintenance_window_duration):
    duration_in_seconds = maintenance_window_duration * 3600
    return six.text_type(duration_in_seconds) + 's'