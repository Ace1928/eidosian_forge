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
def CreateInstanceReferences(holder, igm_ref, instance_names):
    """Creates references to instances in instance group (zonal or regional)."""
    if igm_ref.Collection() == 'compute.instanceGroupManagers':
        instance_refs = []
        for instance in instance_names:
            instance_refs.append(holder.resources.Parse(instance, params={'project': igm_ref.project, 'zone': igm_ref.zone}, collection='compute.instances'))
        return instance_refs
    elif igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
        messages = holder.client.messages
        request = messages.ComputeRegionInstanceGroupManagersListManagedInstancesRequest(instanceGroupManager=igm_ref.Name(), region=igm_ref.region, project=igm_ref.project)
        managed_instances = list_pager.YieldFromList(service=holder.client.apitools_client.regionInstanceGroupManagers, batch_size=500, request=request, method='ListManagedInstances', field='managedInstances')
        instances_to_return = []
        for instance_ref in managed_instances:
            if path_simplifier.Name(instance_ref.instance) in instance_names or instance_ref.instance in instance_names:
                instances_to_return.append(instance_ref.instance)
        return instances_to_return
    else:
        raise ValueError('Unknown reference type {0}'.format(igm_ref.Collection()))