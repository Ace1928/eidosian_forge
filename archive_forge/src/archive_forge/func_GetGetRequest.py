from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.health_checks import exceptions
from googlecloudsdk.command_lib.compute.http_health_checks import flags
from googlecloudsdk.core import log
def GetGetRequest(self, client, http_health_check_ref):
    """Returns a request for fetching the existing HTTP health check."""
    return (client.apitools_client.httpHealthChecks, 'Get', client.messages.ComputeHttpHealthChecksGetRequest(httpHealthCheck=http_health_check_ref.Name(), project=http_health_check_ref.project))