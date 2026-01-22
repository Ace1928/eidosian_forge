from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnetwork_flags
from googlecloudsdk.command_lib.compute.service_attachments import flags
from googlecloudsdk.command_lib.compute.service_attachments import service_attachments_utils
def _GetProjectOrNetwork(self, consumer_limit):
    if consumer_limit.projectIdOrNum is not None:
        return (consumer_limit.projectIdOrNum, consumer_limit.connectionLimit)
    return (consumer_limit.networkUrl, consumer_limit.connectionLimit)