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
def AddSerialConsoleSshKeyArgToParser(parser, positional=False, name=None):
    """Sets up an argument for the serial-console-ssh-key resource."""
    name = 'serial_console_ssh_key' if positional else '--serial-console-ssh-key'
    ssh_key_data = yaml_data.ResourceYAMLData.FromPath('bms.serial_console_ssh_key')
    resource_spec = concepts.ResourceSpec.FromYaml(ssh_key_data.GetData())
    presentation_spec = presentation_specs.ResourcePresentationSpec(name=name, concept_spec=resource_spec, required=True, flag_name_overrides={'region': ''}, group_help='serial_console_ssh_key.')
    return concept_parsers.ConceptParser([presentation_spec]).AddToParser(parser)