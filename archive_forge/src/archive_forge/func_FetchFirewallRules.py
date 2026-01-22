from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.api_lib.compute import base_classes as compute_base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions as command_exceptions
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def FetchFirewallRules():
    """Fetches the firewall rules for the current project.

  Returns:
    A list of firewall rules.
  """
    holder = compute_base_classes.ComputeApiHolder(base.ReleaseTrack.GA)
    client = holder.client
    request_data = lister._Frontend(None, None, lister.GlobalScope([holder.resources.Parse(properties.VALUES.core.project.GetOrFail(), collection='compute.projects')]))
    list_implementation = lister.GlobalLister(client, client.apitools_client.firewalls)
    result = lister.Invoke(request_data, list_implementation)
    return result