from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddSshKeyArgToParser(parser, positional=False, name=None, required=True, plural=False):
    """Sets up an argument for the ssh-key resource."""
    name = 'ssh_key' if positional else '--ssh-key'
    ssh_key_data = yaml_data.ResourceYAMLData.FromPath('bms.ssh_key')
    resource_spec = concepts.ResourceSpec.FromYaml(ssh_key_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name if not plural else f'{name}s', concept_spec=resource_spec, required=required, flag_name_overrides={'region': ''}, group_help='ssh_key.', plural=plural)
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)