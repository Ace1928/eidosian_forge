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
def OutputNamedPortsForGroup(group_ref, compute_client):
    """Gets the request to fetch instance group."""
    compute = compute_client.apitools_client
    if group_ref.Collection() == 'compute.instanceGroups':
        service = compute.instanceGroups
        request = service.GetRequestType('Get')(instanceGroup=group_ref.Name(), zone=group_ref.zone, project=group_ref.project)
    else:
        service = compute.regionInstanceGroups
        request = service.GetRequestType('Get')(instanceGroup=group_ref.Name(), region=group_ref.region, project=group_ref.project)
    results = compute_client.MakeRequests(requests=[(service, 'Get', request)])
    return list(UnwrapResponse(results, 'namedPorts'))