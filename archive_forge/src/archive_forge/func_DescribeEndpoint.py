from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
def DescribeEndpoint(self, name):
    """Calls the GetEndpoint API."""
    get_request = self.messages.NetworksecurityOrganizationsLocationsFirewallEndpointsGetRequest(name=name)
    return self._endpoint_client.Get(get_request)