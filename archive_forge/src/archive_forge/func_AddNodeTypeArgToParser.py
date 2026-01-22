from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddNodeTypeArgToParser(parser, positional=False):
    """Parses node type flag."""
    if positional:
        name = 'node_type'
        flag_name_overrides = None
    else:
        name = '--node-type'
        flag_name_overrides = {'location': ''}
    location_data = yaml_data.ResourceYAMLData.FromPath('vmware.node_type')
    resource_spec = concepts.ResourceSpec.FromYaml(location_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='node_type.', flag_name_overrides=flag_name_overrides)
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)