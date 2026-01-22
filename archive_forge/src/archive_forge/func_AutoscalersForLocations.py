from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import random
import re
import string
import sys
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import auto_healing_utils
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves import range  # pylint: disable=redefined-builtin
def AutoscalersForLocations(zones, regions, client, fail_when_api_not_supported=True):
    """Finds all Autoscalers defined for a given project and locations.

  Args:
    zones: iterable of target zone references
    regions: iterable of target region references
    client: The compute client.
    fail_when_api_not_supported: If true, raise tool exception if API does not
      support autoscaling.

  Returns:
    A list of Autoscaler objects.
  """
    errors = []
    requests = []
    for project, zones in six.iteritems(GroupByProject(zones)):
        requests += lister.FormatListRequests(service=client.apitools_client.autoscalers, project=project, scopes=sorted(set([zone_ref.zone for zone_ref in zones])), scope_name='zone', filter_expr=None)
    if regions:
        if hasattr(client.apitools_client, 'regionAutoscalers'):
            for project, regions in six.iteritems(GroupByProject(regions)):
                requests += lister.FormatListRequests(service=client.apitools_client.regionAutoscalers, project=project, scopes=sorted(set([region_ref.region for region_ref in regions])), scope_name='region', filter_expr=None)
        elif fail_when_api_not_supported:
            errors.append((None, 'API does not support regional autoscaling'))
    autoscalers = client.MakeRequests(requests=requests, errors_to_collect=errors)
    if errors:
        utils.RaiseToolException(errors, error_message='Could not check if the Managed Instance Group is Autoscaled.')
    return autoscalers