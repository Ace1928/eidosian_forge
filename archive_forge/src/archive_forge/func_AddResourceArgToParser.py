from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def AddResourceArgToParser(parser, resource_path, help_text, name=None, required=True):
    """Adds a resource argument in a python command.

  Args:
    parser: the parser for the command.
    resource_path: string, the resource_path which refers to the resources.yaml.
    help_text: string, the help text of the resource argument.
    name: string, the default is the name specified in the resources.yaml file.
    required: boolean, the default is True because in most cases resource arg is
      required.
  """
    resource_yaml_data = yaml_data.ResourceYAMLData.FromPath(resource_path)
    resource_spec = concepts.ResourceSpec.FromYaml(resource_yaml_data.GetData())
    concept_parsers.ConceptParser.ForResource(name=name if name else resource_yaml_data.GetArgName(), resource_spec=resource_spec, group_help=help_text, required=required).AddToParser(parser)