from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def RemoveAuthorizedNetworksFlag(unused_domain_ref, args, patch_request):
    """Removes authorized networks from domain."""
    if args.IsSpecified('remove_authorized_networks'):
        ans = [an for an in patch_request.domain.authorizedNetworks if an not in args.remove_authorized_networks]
        ans = sorted(set(ans))
        patch_request.domain.authorizedNetworks = ans
        AddFieldToUpdateMask('authorized_networks', patch_request)
    return patch_request