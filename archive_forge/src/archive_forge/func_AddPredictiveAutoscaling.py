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
def AddPredictiveAutoscaling(parser, standard=True):
    """Add Predictive autoscaling arguments to the parser."""
    choices = {'none': '(Default) No predictions are made when calculating the number of VM\ninstances.\n', 'optimize-availability': 'Predictive autoscaling predicts the future values of the\nscaling metric and scales the group in advance to ensure that new\nVM instances are ready in time to cover the predicted peak.\n'}
    if standard:
        choices['standard'] = '\n    Standard predictive autoscaling  predicts the future values of\n    the scaling metric and then scales the group to ensure that new VM\n    instances are ready in time to cover the predicted peak.'
    parser.add_argument('--cpu-utilization-predictive-method', choices=choices, help='\n      Indicates whether to use a predictive algorithm when scaling based on\n      CPU.')