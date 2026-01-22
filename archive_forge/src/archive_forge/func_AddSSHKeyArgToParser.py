from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddSSHKeyArgToParser(parser, positional=False):
    """Sets up an argument for the image resource."""
    if positional:
        name = 'ssh_key'
    else:
        name = '--ssh_key'
    ssh_key_data = yaml_data.ResourceYAMLData.FromPath('mps.power_ssh_key')
    resource_spec = concepts.ResourceSpec.FromYaml(ssh_key_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, group_help='ssh-key.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)