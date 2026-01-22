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
def AddScheduledAutoscalingConfigurationArguments(arg_group):
    """Add arguments that are common to adding or modifying a scaling schedule."""
    arg_group.add_argument('--schedule-cron', metavar='CRON_EXPRESSION', help="        Start time of the scaling schedule in cron format.\n\n        This is when the autoscaler starts creating new VMs, if the group's\n        current size is less than the minimum required instances. Set the start\n        time to allow enough time for new VMs to boot and initialize. For\n        example if your workload takes 10 minutes from VM creation to start\n        serving then set the start time 10 minutes earlier than the time you\n        need VMs to be ready.\n        ")
    arg_group.add_argument('--schedule-duration-sec', metavar='DURATION', type=arg_parsers.BoundedInt(300, sys.maxsize), help="        How long should the scaling schedule be active, measured in seconds.\n\n        Minimum duration is 5 minutes. A scaling schedule is active from its\n        start time and for its configured duration. During this time, the\n        autoscaler scales the group to have at least as many VMs as defined by\n        the minimum required instances. After the configured duration, if there\n        is no need to maintain capacity, the autoscaler starts removing\n        instances after the usual stabilization period and after scale-in\n        controls (if configured). For more information, see\n        [Delays in scaling in](https://cloud.google.com/compute/docs/autoscaler/understanding-autoscaler-decisions#delays_in_scaling_in) and [Scale-in controls](https://cloud.google.com/compute/docs/autoscaler/understanding-autoscaler-decisions#scale-in_controls).\n        This ensures you don't accidentally lose capacity immediately after\n        the scaling schedule ends.\n        ")
    arg_group.add_argument('--schedule-min-required-replicas', metavar='MIN_REQUIRED_REPLICAS', type=arg_parsers.BoundedInt(0, sys.maxsize), help='        How many VMs the autoscaler should provision for the duration of this\n        scaling schedule.\n\n        Autoscaler provides at least this number of instances when the scaling\n        schedule is active. A managed instance group can have more VMs if there\n        are other scaling schedules active with more required instances or if\n        another signal (for example, scaling based on CPU) requires more\n        instances to meet its target.\n\n        This configuration does not change autoscaling minimum and maximum\n        instance limits which are always in effect. Autoscaler does not create\n        more than the maximum number of instances configured for a group.\n        ')
    arg_group.add_argument('--schedule-time-zone', metavar='TIME_ZONE', help="        Name of the timezone that the scaling schedule's start time is in.\n\n        It should be provided as a name from the IANA tz database (for\n        example Europe/Paris or UTC). It automatically adjusts for daylight\n        savings time (DST). If no time zone is provided, UTC is used as a\n        default.\n\n        See https://en.wikipedia.org/wiki/List_of_tz_database_time_zones for\n        the list of valid timezones.\n        ")
    arg_group.add_argument('--schedule-description', metavar='DESCRIPTION', help='A verbose description of the scaling schedule.')