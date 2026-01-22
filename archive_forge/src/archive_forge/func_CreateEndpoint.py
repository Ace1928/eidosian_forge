from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def CreateEndpoint(self, name, parent, description, billing_project_id, endpoint_type=None, target_firewall_attachment=None, labels=None):
    """Calls the CreateEndpoint API."""
    third_party_endpoint_settings = self._ParseThirdPartyEndpointSettings(target_firewall_attachment)
    if endpoint_type is not None:
        endpoint = self.messages.FirewallEndpoint(labels=labels, type=self._ParseEndpointType(endpoint_type), thirdPartyEndpointSettings=third_party_endpoint_settings, description=description, billingProjectId=billing_project_id)
    else:
        endpoint = self.messages.FirewallEndpoint(labels=labels, description=description, billingProjectId=billing_project_id)
    create_request = self.messages.NetworksecurityOrganizationsLocationsFirewallEndpointsCreateRequest(firewallEndpoint=endpoint, firewallEndpointId=name, parent=parent)
    return self._endpoint_client.Create(create_request)