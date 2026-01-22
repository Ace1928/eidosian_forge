from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_ALPHA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_BETA
from googlecloudsdk.command_lib.composer.flags import CONNECTION_TYPE_FLAG_GA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_ALPHA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_BETA
from googlecloudsdk.command_lib.composer.flags import ENVIRONMENT_SIZE_GA
def ConvertToTypeEnum(type_enum, airflow_executor_type):
    """Converts airflow executor type string to enum.

  Args:
    type_enum: AirflowExecutorTypeValueValuesEnum, executor type enum value.
    airflow_executor_type: string, executor type string value.

  Returns:
    AirflowExecutorTypeValueValuesEnum: the executor type enum value.
  """
    return type_enum(airflow_executor_type)