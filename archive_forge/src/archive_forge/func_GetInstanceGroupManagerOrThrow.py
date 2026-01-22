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
def GetInstanceGroupManagerOrThrow(igm_ref, client):
    """Retrieves the given Instance Group Manager if possible.

  Args:
    igm_ref: reference to the Instance Group Manager.
    client: The compute client.

  Returns:
    Instance Group Manager object.
  """
    if hasattr(igm_ref, 'region'):
        service = client.apitools_client.regionInstanceGroupManagers
        request_type = service.GetRequestType('Get')
    if hasattr(igm_ref, 'zone'):
        service = client.apitools_client.instanceGroupManagers
        request_type = service.GetRequestType('Get')
    request = request_type(**igm_ref.AsDict())
    errors = []
    igm_details = client.MakeRequests([(service, 'Get', request)], errors_to_collect=errors)
    if errors or len(igm_details) != 1:
        utils.RaiseException(errors, ResourceNotFoundException, error_message='Could not fetch resource:')
    return igm_details[0]