from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
def GetSetNamedPortsRequestForGroup(compute_client, group_ref, ports):
    """Returns a request to get named ports and service to send request.

  Args:
    compute_client: GCE API client,
    group_ref: reference to instance group (zonal or regional),
    ports: list of named ports to set

  Returns:
    request, message to send in order to set named ports on instance group,
    service, service where request should be sent
      - regionInstanceGroups for regional groups
      - instanceGroups for zonal groups
  """
    compute = compute_client.apitools_client
    messages = compute_client.messages
    fingerprint = _GetGroupFingerprint(compute_client, group_ref)
    if IsZonalGroup(group_ref):
        request_body = messages.InstanceGroupsSetNamedPortsRequest(fingerprint=fingerprint, namedPorts=ports)
        return (messages.ComputeInstanceGroupsSetNamedPortsRequest(instanceGroup=group_ref.Name(), instanceGroupsSetNamedPortsRequest=request_body, zone=group_ref.zone, project=group_ref.project), compute.instanceGroups)
    else:
        request_body = messages.RegionInstanceGroupsSetNamedPortsRequest(fingerprint=fingerprint, namedPorts=ports)
        return (messages.ComputeRegionInstanceGroupsSetNamedPortsRequest(instanceGroup=group_ref.Name(), regionInstanceGroupsSetNamedPortsRequest=request_body, region=group_ref.region, project=group_ref.project), compute.regionInstanceGroups)