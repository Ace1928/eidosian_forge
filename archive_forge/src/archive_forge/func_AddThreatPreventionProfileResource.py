from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.network_security.security_profile_groups import spg_api
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddThreatPreventionProfileResource(parser, release_track, help_text='Path to Security Profile resource.', required=False):
    """Adds Security Profile resource."""
    api_version = spg_api.GetApiVersion(release_track)
    collection_info = resources.REGISTRY.Clone().GetCollectionInfo(_SECURITY_PROFILE_GROUP_RESOURCE_COLLECTION, api_version)
    resource_spec = concepts.ResourceSpec(_SECURITY_PROFILE_RESOURCE_COLLECTION, 'Security Profile', api_version=api_version, organizationsId=concepts.ResourceParameterAttributeConfig('security-profile-organization', 'Organization ID of the Security Profile.', parameter_name='organizationsId', fallthroughs=[deps.ArgFallthrough('--organization'), deps.FullySpecifiedAnchorFallthrough([deps.ArgFallthrough(_SECURITY_PROFILE_GROUP_RESOURCE_COLLECTION)], collection_info, 'organizationsId')]), locationsId=concepts.ResourceParameterAttributeConfig('security-profile-location', '\n          Location of the {resource}.\n          NOTE: Only `global` security profiles are supported.\n          ', parameter_name='locationsId', fallthroughs=[deps.ArgFallthrough('--location'), deps.FullySpecifiedAnchorFallthrough([deps.ArgFallthrough(_SECURITY_PROFILE_GROUP_RESOURCE_COLLECTION)], collection_info, 'locationsId')]), securityProfilesId=concepts.ResourceParameterAttributeConfig('security_profile', 'Name of security profile {resource}.', parameter_name='securityProfilesId'))
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=_THREAT_PREVENTION_PROFILE_RESOURCE_NAME, concept_spec=resource_spec, required=required, group_help=help_text)
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)