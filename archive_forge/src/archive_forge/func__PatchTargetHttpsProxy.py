from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import target_proxies_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.certificate_manager import resource_args
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import reference_utils
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.ssl_certificates import flags as ssl_certificates_flags
from googlecloudsdk.command_lib.compute.ssl_policies import flags as ssl_policies_flags
from googlecloudsdk.command_lib.compute.target_https_proxies import flags
from googlecloudsdk.command_lib.compute.target_https_proxies import target_https_proxies_utils
from googlecloudsdk.command_lib.compute.url_maps import flags as url_map_flags
from googlecloudsdk.command_lib.network_security import resource_args as ns_resource_args
def _PatchTargetHttpsProxy(client, proxy_ref, new_resource, cleared_fields):
    """Patches the target HTTPS proxy."""
    requests = []
    if target_https_proxies_utils.IsRegionalTargetHttpsProxiesRef(proxy_ref):
        requests.append((client.apitools_client.regionTargetHttpsProxies, 'Patch', client.messages.ComputeRegionTargetHttpsProxiesPatchRequest(project=proxy_ref.project, region=proxy_ref.region, targetHttpsProxy=proxy_ref.Name(), targetHttpsProxyResource=new_resource)))
    else:
        requests.append((client.apitools_client.targetHttpsProxies, 'Patch', client.messages.ComputeTargetHttpsProxiesPatchRequest(project=proxy_ref.project, targetHttpsProxy=proxy_ref.Name(), targetHttpsProxyResource=new_resource)))
    with client.apitools_client.IncludeFields(cleared_fields):
        return client.MakeRequests(requests)