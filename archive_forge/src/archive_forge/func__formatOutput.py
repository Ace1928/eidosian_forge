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
def _formatOutput(record):
    output = []
    header = k8s_util.BuildHeader(record)
    status = ExecutionPrinter.TransformStatus(record)
    labels = k8s_util.GetLabels(record.labels)
    spec = ExecutionPrinter.TransformSpec(record)
    ready_message = k8s_util.FormatReadyMessage(record)
    if header:
        output.append(header)
    if status:
        output.append(status)
    output.append(' ')
    if labels:
        output.append(labels)
        output.append(' ')
    if spec:
        output.append(spec)
    if ready_message:
        output.append(ready_message)
    return output