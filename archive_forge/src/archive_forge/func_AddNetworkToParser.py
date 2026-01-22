from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddNetworkToParser(parser, positional=False):
    """Sets up an argument for the VMware Engine network resource."""
    name = '--vmware-engine-network'
    if positional:
        name = 'vmware_engine_network'
    network_data = yaml_data.ResourceYAMLData.FromPath('vmware.networks.vmware_engine_network')
    resource_spec = concepts.ResourceSpec.FromYaml(network_data.GetData())
    if positional:
        presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='vmware_engine_network.')
    else:
        presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='vmware_engine_network.', flag_name_overrides={'location': '--network-location'})
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)