from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
from googlecloudsdk.command_lib.run.printers import container_and_volume_printer_util as container_util
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.command_lib.util import time_util
from googlecloudsdk.core.resource import custom_printer_base as cp
@staticmethod
def _elapsedTime(start, end):
    duration = datetime.timedelta(seconds=time_util.Strptime(end) - time_util.Strptime(start)).seconds
    hours = duration // 3600
    duration = duration % 3600
    minutes = duration // 60
    seconds = duration % 60
    if hours > 0:
        return '{} and {}'.format(_PluralizedWord('hour', hours), _PluralizedWord('minute', minutes))
    if minutes > 0:
        return '{} and {}'.format(_PluralizedWord('minute', minutes), _PluralizedWord('second', seconds))
    return _PluralizedWord('second', seconds)