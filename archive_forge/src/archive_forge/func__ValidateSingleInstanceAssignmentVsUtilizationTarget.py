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
def _ValidateSingleInstanceAssignmentVsUtilizationTarget(args):
    if args.IsSpecified('stackdriver_metric_single_instance_assignment'):
        potential_conflicting = ['stackdriver_metric_utilization_target', 'stackdriver_metric_utilization_target_type']
        conflicting = [f for f in potential_conflicting if args.IsSpecified(f)]
        if any(conflicting):
            assignment_flag = '--stackdriver-metric-single-instance-assignment'
            conflicting_flags = ['[--{}]'.format(f.replace('_', '-')) for f in conflicting]
            raise calliope_exceptions.ConflictingArgumentsException(assignment_flag, 'You cannot use any of {} with `{}`'.format(conflicting_flags, assignment_flag))