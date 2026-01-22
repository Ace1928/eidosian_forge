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
def GetModeFlag():
    return base.ChoiceArgument('--mode', {'on': 'Permits autoscaling to scale out and in (default for new autoscalers).', 'only-scale-out': 'Permits autoscaling to scale only out and not in.', 'only-up': '\n              (DEPRECATED) Permits autoscaling to scale only out and not in.\n\n              Value `only-up` is deprecated. Use `--mode only-scale-out`\n              instead.\n          ', 'off': 'Turns off autoscaling, while keeping the new configuration.'}, help_str="          Set the mode of an autoscaler for a managed instance group.\n\n          You can turn off or restrict a group's autoscaler activities without\n          affecting your autoscaler configuration. The autoscaler configuration\n          persists while the activities are turned off or restricted, and the\n          activities resume when the autoscaler is turned on again or when the\n          restrictions are lifted.\n      ")