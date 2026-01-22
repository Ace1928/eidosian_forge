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
def TransformStatus(record):
    if record.status is None:
        return ''
    lines = []
    if record.ready_condition['status'] is None:
        lines.append('{} currently running'.format(_PluralizedWord('task', record.status.runningCount)))
    lines.append('{} completed successfully'.format(_PluralizedWord('task', record.status.succeededCount)))
    if record.status.failedCount is not None and record.status.failedCount > 0:
        lines.append('{} failed to complete'.format(_PluralizedWord('task', record.status.failedCount)))
    if record.status.cancelledCount is not None and record.status.cancelledCount > 0:
        lines.append('{} cancelled'.format(_PluralizedWord('task', record.status.cancelledCount)))
    if record.status.completionTime is not None and record.creation_timestamp is not None:
        lines.append('Elapsed time: ' + ExecutionPrinter._elapsedTime(record.creation_timestamp, record.status.completionTime))
    if record.status.logUri is not None:
        lines.append(' ')
        lines.append('Log URI: {}'.format(record.status.logUri))
    return cp.Lines(lines)