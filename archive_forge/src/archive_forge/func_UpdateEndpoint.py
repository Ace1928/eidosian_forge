from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def UpdateEndpoint(self, name, description, update_mask, labels=None, billing_project_id=None):
    """Calls the UpdateEndpoint API.

    Args:
      name: str, full name of the firewall endpoint.
      description: str, description of the firewall endpoint.
      update_mask: str, comma separated list of fields to update.
      labels: LabelsValue, labels for the firewall endpoint.
      billing_project_id: str, billing project ID.
    Returns:
      Operation ref to track the long-running process.
    """
    endpoint = self.messages.FirewallEndpoint(labels=labels, description=description, billingProjectId=billing_project_id)
    update_request = self.messages.NetworksecurityOrganizationsLocationsFirewallEndpointsPatchRequest(name=name, firewallEndpoint=endpoint, updateMask=update_mask)
    return self._endpoint_client.Patch(update_request)