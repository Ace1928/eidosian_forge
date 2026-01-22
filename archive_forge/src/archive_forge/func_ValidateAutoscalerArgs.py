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
def ValidateAutoscalerArgs(args):
    """Validates args."""
    if args.min_num_replicas and args.max_num_replicas:
        if args.min_num_replicas > args.max_num_replicas:
            raise calliope_exceptions.InvalidArgumentException('--max-num-replicas', "can't be less than min num replicas.")
    if args.custom_metric_utilization:
        for custom_metric_utilization in args.custom_metric_utilization:
            for field in ('utilization-target', 'metric', 'utilization-target-type'):
                if field not in custom_metric_utilization:
                    raise calliope_exceptions.InvalidArgumentException('--custom-metric-utilization', field + ' not present.')
            if custom_metric_utilization['utilization-target'] < 0:
                raise calliope_exceptions.InvalidArgumentException('--custom-metric-utilization utilization-target', 'less than 0.')