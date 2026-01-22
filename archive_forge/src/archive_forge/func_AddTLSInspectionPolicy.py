from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.network_security.firewall_endpoints import activation_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import resources
def AddTLSInspectionPolicy(release_track, parser, help_text='Path to TLS Inspection Policy configuration to use for intercepting TLS-encrypted traffic in this network.'):
    """Adds TLS Inspection Policy resource."""
    api_version = activation_api.GetApiVersion(release_track)
    collection_info = resources.REGISTRY.Clone().GetCollectionInfo(ASSOCIATION_RESOURCE_COLLECTION, api_version)
    resource_spec = concepts.ResourceSpec(TLS_INSPECTION_POLICY_RESOURCE_COLLECTION, 'TLS Inspection Policy', api_version=api_version, projectsId=concepts.ResourceParameterAttributeConfig('tls-inspection-policy-project', 'Project of the {resource}.', parameter_name='projectsId', fallthroughs=[deps.ArgFallthrough('--project'), deps.FullySpecifiedAnchorFallthrough([deps.ArgFallthrough(ASSOCIATION_RESOURCE_NAME)], collection_info, 'projectsId')]), locationsId=concepts.ResourceParameterAttributeConfig('tls-inspection-policy-region', '\n          Region of the {resource}.\n          NOTE: TLS Inspection Policy needs to be\n          in the same region as Firewall Plus endpoint resource.\n          ', parameter_name='locationsId'), tlsInspectionPoliciesId=concepts.ResourceParameterAttributeConfig('tls_inspection_policy', 'Name of the {resource}', parameter_name='tlsInspectionPoliciesId'))
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=TLS_INSPECTION_POLICY_RESOURCE_NAME, concept_spec=resource_spec, required=False, group_help=help_text)
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)