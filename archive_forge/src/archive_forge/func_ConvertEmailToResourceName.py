from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.organizations import org_utils
import six
def ConvertEmailToResourceName(version, email, arg_name):
    """Convert email to resource name.

  Args:
    version: Release track information
    email: group email
    arg_name: argument/parameter name

  Returns:
    Group Id (e.g. groups/11zu0gzc3tkdgn2)

  """
    try:
        return ci_client.LookupGroupName(version, email).name
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError):
        error_msg = 'There is no such a group associated with the specified argument:' + email
        raise exceptions.InvalidArgumentException(arg_name, error_msg)