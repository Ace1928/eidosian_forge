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
def AdjustAutoscalerNameForCreation(autoscaler_resource, igm_ref):
    """Set name of autoscaler o be created.

  If autoscaler name is not None it wNone ill be used as a prefix of name of the
  autoscaler to be created. Prefix may be shortened so that the name fits below
  length limit. Name prefix is followed by '-' character and four
  random letters.

  Args:
    autoscaler_resource: Autoscaler resource to be created.
    igm_ref: reference to Instance Group Manager targeted by the Autoscaler.
  """
    if autoscaler_resource.name is None:
        autoscaler_resource.name = igm_ref.Name()
    trimmed_name = autoscaler_resource.name[0:_MAX_AUTOSCALER_NAME_LENGTH - _NUM_RANDOM_CHARACTERS_IN_AS_NAME - 1]
    random_characters = [random.choice(string.ascii_lowercase + string.digits) for _ in range(_NUM_RANDOM_CHARACTERS_IN_AS_NAME)]
    random_suffix = ''.join(random_characters)
    new_name = '{0}-{1}'.format(trimmed_name, random_suffix)
    autoscaler_resource.name = new_name