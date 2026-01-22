from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddPrivatecloudArgToParser(parser, positional=False):
    """Sets up an argument for the privatecloud resource."""
    if positional:
        name = 'privatecloud'
    else:
        name = '--privatecloud'
    privatecloud_data = yaml_data.ResourceYAMLData.FromPath('vmware.sddc.privatecloud')
    resource_spec = concepts.ResourceSpec.FromYaml(privatecloud_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='privatecloud.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)