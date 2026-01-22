from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.target_grpc_proxies import flags
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def _Patch(client, target_grpc_proxy_ref, target_grpc_proxy):
    """Make target gRPC proxy patch request."""
    request = client.messages.ComputeTargetGrpcProxiesPatchRequest(project=target_grpc_proxy_ref.project, targetGrpcProxy=target_grpc_proxy_ref.Name(), targetGrpcProxyResource=target_grpc_proxy)
    collection = client.apitools_client.targetGrpcProxies
    return client.MakeRequests([(collection, 'Patch', request)])[0]