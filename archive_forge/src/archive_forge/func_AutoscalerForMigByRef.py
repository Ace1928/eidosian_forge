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
def AutoscalerForMigByRef(client, resources, igm_ref):
    """Returns autoscaler targeting given instance group manager.

  Args:
    client: a GCE client
    resources: a GCE resource registry
    igm_ref: reference to instance group manager

  Returns:
    Autoscaler message with autoscaler targeting the IGM refferenced by
    igm_ref or None if there isn't one.
  Raises:
    ValueError: if instance group manager collection path is unknown
  """
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        scope_type = 'zone'
        location = CreateZoneRef(resources, igm_ref)
        zones, regions = ([location], None)
    elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
        scope_type = 'region'
        location = CreateRegionRef(resources, igm_ref)
        zones, regions = (None, [location])
    else:
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))
    autoscalers = AutoscalersForLocations(regions=regions, zones=zones, client=client)
    return AutoscalerForMig(mig_name=igm_ref.Name(), autoscalers=autoscalers, location=location, scope_type=scope_type)