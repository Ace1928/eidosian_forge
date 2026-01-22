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
def GetExecutionEnv(record):
    execution_env_value = k8s_util.GetExecutionEnvironment(record)
    if execution_env_value in EXECUTION_ENV_VALS:
        return EXECUTION_ENV_VALS[execution_env_value]
    return execution_env_value