from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import datetime
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def _CustomMachineTypeString(cpu, memory_mib):
    """Creates a custom machine type from the CPU and memory specs.

  Args:
    cpu: the number of cpu desired for the custom machine type
    memory_mib: the amount of ram desired in MiB for the custom machine type
      instance

  Returns:
    The custom machine type name for the 'instance create' call
  """
    machine_type = 'db-custom-{0}-{1}'.format(cpu, memory_mib)
    return machine_type