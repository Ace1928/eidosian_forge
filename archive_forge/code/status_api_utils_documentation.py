from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util as fleet_api_util
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
import six
Auto-page through the responses if the next page token is not empty and returns a list of all resources.

  Args:
    client_fn: Function specific to the endpoint
    request: Request object specific to the endpoint
    project_id: Project id that will be used in populating the request object
    resource_collector: Function to be used for retrieving the relevant field
      from the response

  Returns:
    List of all resources specific to the endpoint
  