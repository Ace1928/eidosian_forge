from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.network_attachments import flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as subnetwork_flags
def _GetOldResource(self, client, network_attachment_ref):
    """Returns the existing NetworkAttachment resource."""
    request = client.messages.ComputeNetworkAttachmentsGetRequest(**network_attachment_ref.AsDict())
    collection = client.apitools_client.networkAttachments
    return client.MakeRequests([(collection, 'Get', request)])[0]