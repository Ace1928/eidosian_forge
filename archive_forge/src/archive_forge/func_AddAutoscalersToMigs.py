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
def AddAutoscalersToMigs(migs_iterator, client, resources, fail_when_api_not_supported=True):
    """Add Autoscaler to each IGM object if autoscaling is enabled for it."""

    def ParseZone(zone_link):
        return resources.Parse(zone_link, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.zones')

    def ParseRegion(region_link):
        return resources.Parse(region_link, params={'project': properties.VALUES.core.project.GetOrFail}, collection='compute.regions')
    migs = list(migs_iterator)
    zones = set([ParseZone(mig['zone']) for mig in migs if 'zone' in mig])
    regions = set([ParseRegion(mig['region']) for mig in migs if 'region' in mig])
    autoscalers = {}
    all_autoscalers = AutoscalersForLocations(zones=zones, regions=regions, client=client, fail_when_api_not_supported=fail_when_api_not_supported)
    for location in list(zones) + list(regions):
        autoscalers[location.Name()] = []
    for autoscaler in all_autoscalers:
        autoscaler_scope = None
        if autoscaler.zone is not None:
            autoscaler_scope = ParseZone(autoscaler.zone)
        if hasattr(autoscaler, 'region') and autoscaler.region is not None:
            autoscaler_scope = ParseRegion(autoscaler.region)
        if autoscaler_scope is not None:
            autoscalers.setdefault(autoscaler_scope.Name(), [])
            autoscalers[autoscaler_scope.Name()].append(autoscaler)
    for mig in migs:
        location = None
        scope_type = None
        if 'region' in mig:
            location = ParseRegion(mig['region'])
            scope_type = 'region'
        elif 'zone' in mig:
            location = ParseZone(mig['zone'])
            scope_type = 'zone'
        autoscaler = None
        if location and scope_type:
            autoscaler = AutoscalerForMig(mig_name=mig['name'], autoscalers=autoscalers[location.Name()], location=location, scope_type=scope_type)
        if autoscaler:
            mig['autoscaler'] = autoscaler
        yield mig