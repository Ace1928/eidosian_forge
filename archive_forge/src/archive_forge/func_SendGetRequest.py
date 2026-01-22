from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import scope as compute_scope
def SendGetRequest(client, target_http_proxy_ref):
    """Send Url Maps get request."""
    if target_http_proxy_ref.Collection() == 'compute.regionTargetHttpProxies':
        return client.apitools_client.regionTargetHttpProxies.Get(client.messages.ComputeRegionTargetHttpProxiesGetRequest(**target_http_proxy_ref.AsDict()))
    return client.apitools_client.targetHttpProxies.Get(client.messages.ComputeTargetHttpProxiesGetRequest(**target_http_proxy_ref.AsDict()))