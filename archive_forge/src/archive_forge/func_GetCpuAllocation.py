from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.command_lib.run.printers import container_and_volume_printer_util as container_util
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.core.resource import custom_printer_base as cp
@staticmethod
def GetCpuAllocation(record):
    cpu_throttled = record.annotations.get(container_resource.CPU_THROTTLE_ANNOTATION)
    if not cpu_throttled:
        return ''
    elif cpu_throttled.lower() == 'false':
        return CPU_ALWAYS_ALLOCATED_MESSAGE
    else:
        return CPU_THROTTLED_MESSAGE