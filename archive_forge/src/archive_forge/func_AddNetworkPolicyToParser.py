from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddNetworkPolicyToParser(parser, positional=False):
    """Sets up an argument for the VMware Engine network policy resource."""
    name = '--network-policy'
    if positional:
        name = 'network_policy'
    network_policy_data = yaml_data.ResourceYAMLData.FromPath('vmware.network_policies.network_policy')
    resource_spec = concepts.ResourceSpec.FromYaml(network_policy_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='network_policy.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)