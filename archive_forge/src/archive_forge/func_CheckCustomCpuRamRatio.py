from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core.util import times
import six
def CheckCustomCpuRamRatio(compute_client, project, zone, machine_type_name):
    """Checks that the CPU and memory ratio is a supported custom instance type.

  Args:
    compute_client: GCE API client,
    project: a project,
    zone: the zone of the instance(s) being created,
    machine_type_name: The machine type of the instance being created.

  Returns:
    Nothing. Function acts as a bound checker, and will raise an exception from
      within the function if needed.

  Raises:
    utils.RaiseToolException if a custom machine type ratio is out of bounds.
  """
    messages = compute_client.messages
    compute = compute_client.apitools_client
    if 'custom' in machine_type_name:
        mt_get_pb = messages.ComputeMachineTypesGetRequest(machineType=machine_type_name, project=project, zone=zone)
        mt_get_reqs = [(compute.machineTypes, 'Get', mt_get_pb)]
        errors = []
        _ = list(compute_client.MakeRequests(requests=mt_get_reqs, errors_to_collect=errors))
        if errors:
            utils.RaiseToolException(errors, error_message='Could not fetch machine type:')