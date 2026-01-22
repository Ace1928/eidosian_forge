from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import target_proxies_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.target_http_proxies import flags
from googlecloudsdk.command_lib.compute.target_http_proxies import target_http_proxies_utils
from googlecloudsdk.command_lib.compute.url_maps import flags as url_map_flags
def _GetGlobalTargetHttpProxy(client, proxy_ref):
    """Retrieves the Global target HTTP proxy."""
    requests = []
    requests.append((client.apitools_client.targetHttpProxies, 'Get', client.messages.ComputeTargetHttpProxiesGetRequest(project=proxy_ref.project, targetHttpProxy=proxy_ref.Name())))
    res = client.MakeRequests(requests)
    return res[0]