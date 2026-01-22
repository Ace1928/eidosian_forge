from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import partner_metadata_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute.instances import flags
def _make_patch_partner_metadata_request(self, client, instance_ref, args):
    partner_metadata_dict = partner_metadata_utils.CreatePartnerMetadataDict(args)
    partner_metadata_message = partner_metadata_utils.ConvertPartnerMetadataDictToMessage(partner_metadata_dict)
    return (client.apitools_client.instances, 'PatchPartnerMetadata', client.messages.ComputeInstancesPatchPartnerMetadataRequest(partnerMetadata=client.messages.PartnerMetadata(partnerMetadata=partner_metadata_message), **instance_ref.AsDict()))