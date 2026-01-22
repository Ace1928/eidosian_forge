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
def AutoscalerForMig(mig_name, autoscalers, location, scope_type):
    """Finds Autoscaler targeting given IGM.

  Args:
    mig_name: Name of MIG targeted by Autoscaler.
    autoscalers: A list of Autoscalers to search among.
    location: Target location reference.
    scope_type: Target scope type.

  Returns:
    Autoscaler object for autoscaling the given Instance Group Manager or None
    when such Autoscaler does not exist.
  Raises:
    InvalidArgumentError: if more than one autoscaler provided
  """
    autoscalers = AutoscalersForMigs([(mig_name, scope_type, location)], autoscalers)
    if autoscalers:
        if len(autoscalers) == 1:
            return autoscalers[0]
        else:
            raise InvalidArgumentError('More than one Autoscaler with given target.')
    return None