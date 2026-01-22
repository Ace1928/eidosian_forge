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
def _AddServerTLSPolicyArguments(parser):
    """Adds all Server TLS Policy-related arguments."""
    server_tls_group = parser.add_mutually_exclusive_group()
    ns_resource_args.GetServerTlsPolicyResourceArg('to attach', name='server-tls-policy', group=server_tls_group, region_fallthrough=True).AddToParser(server_tls_group)
    ns_resource_args.GetClearServerTLSPolicyForHttpsProxy().AddToParser(server_tls_group)