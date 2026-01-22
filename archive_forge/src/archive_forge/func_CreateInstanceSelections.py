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
def CreateInstanceSelections(args, messages, igm_resource):
    """Build a list of InstanceSelection from the given flags."""
    instance_selections = []
    if args.IsKnownAndSpecified('remove_instance_selections_all') or args.IsKnownAndSpecified('remove_instance_selections'):
        RegisterInstanceSelectionsPatchEncoders(messages)
        existing_instance_selection_names = _GetExistingInstanceSelectionNames(igm_resource)
        if args.IsKnownAndSpecified('remove_instance_selections_all'):
            for instance_selection_name in existing_instance_selection_names:
                instance_selections.append(messages.InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue.AdditionalProperty(key=instance_selection_name, value=None))
        elif args.IsKnownAndSpecified('remove_instance_selections'):
            for instance_selection_name in args.remove_instance_selections:
                if instance_selection_name not in existing_instance_selection_names:
                    continue
                if any((instance_selection.key == instance_selection_name for instance_selection in instance_selections)):
                    continue
                instance_selections.append(messages.InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue.AdditionalProperty(key=instance_selection_name, value=None))
    if args.IsKnownAndSpecified('instance_selection_machine_types'):
        _AddInstanceSelection(messages, instance_selections, 'instance-selection-1', args.instance_selection_machine_types, 1)
    if args.IsKnownAndSpecified('instance_selection'):
        for instance_selection in args.instance_selection:
            if 'name' not in instance_selection:
                raise InvalidArgumentError('Missing instance selection name.')
            name = instance_selection['name'][0]
            if 'machine-type' not in instance_selection or not instance_selection['machine-type']:
                raise InvalidArgumentError('Missing machine type in instance selection.')
            machine_types = instance_selection['machine-type']
            rank = None
            if 'rank' in instance_selection:
                rank = instance_selection['rank'][0]
                if not rank.isdigit():
                    raise InvalidArgumentError('Invalid value for rank in instance selection.')
                rank = int(rank)
            _AddInstanceSelection(messages, instance_selections, name, machine_types, rank)
    if not instance_selections:
        return None
    return messages.InstanceGroupManagerInstanceFlexibilityPolicy.InstanceSelectionsValue(additionalProperties=instance_selections)