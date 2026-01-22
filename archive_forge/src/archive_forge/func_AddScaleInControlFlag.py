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
def AddScaleInControlFlag(parser, include_clear=False):
    """Adds scale-in-control flags to the given parser."""
    arg_group = parser
    if include_clear:
        arg_group = parser.add_group(mutex=True)
        arg_group.add_argument('--clear-scale-in-control', action='store_true', help='          If specified, the scale-in-control field will be cleared. Using this\n          flag will remove any configuration set by `--scale-in-control` flag.\n        ')
    arg_group.add_argument('--scale-in-control', type=arg_parsers.ArgDict(spec={'max-scaled-in-replicas': str, 'max-scaled-in-replicas-percent': str, 'time-window': int}), help="        Configuration that allows slower scale in so that even if Autoscaler\n        recommends an abrupt scale in of a managed instance group, it will be\n        throttled as specified by the parameters.\n\n        *max-scaled-in-replicas*::: Maximum allowed number of VMs that can be\n        deducted from the peak recommendation during the window. Possibly all\n        these VMs can be deleted at once so the application needs to be prepared\n        to lose that many VMs in one step. Mutually exclusive with\n        'max-scaled-in-replicas-percent'.\n\n        *max-scaled-in-replicas-percent*::: Maximum allowed percent of VMs\n        that can be deducted from the peak recommendation during the window.\n        Possibly all these VMs can be deleted at once so the application needs\n        to be prepared to lose that many VMs in one step. Mutually exclusive\n        with  'max-scaled-in-replicas'.\n\n        *time-window*::: How long back autoscaling should look when computing\n        recommendations. The autoscaler will not resize below the maximum\n        allowed deduction subtracted from the peak size observed in this\n        period. Measured in seconds.\n        ")