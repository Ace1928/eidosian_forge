from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def AddBasicAndCustomSpecArgs(parser, api_version):
    """Add args for basic and custom specs (grouped together)."""
    basic_level_help_text = 'Path to a file containing a list of basic access level conditions.\n\nAn access level condition file is a YAML-formatted list of conditions,which are YAML objects representing a Condition as described in the API reference. For example:\n\n    ```\n     - ipSubnetworks:\n       - 162.222.181.197/24\n       - 2001:db8::/48\n     - members:\n       - user:user@example.com\n    ```'
    custom_level_help_text = 'Path to a file representing an expression for an access level.\n\nThe expression is in the Common Expression Langague (CEL) format.For example:\n\n    ```\n     expression: "origin.region_code in [\'US\', \'CA\']"\n    ```'
    basic_level_spec_arg = base.Argument('--basic-level-spec', help=basic_level_help_text, type=ParseBasicLevelConditions(api_version))
    basic_level_combine_arg = GetCombineFunctionEnumMapper(api_version=api_version).choice_arg
    basic_level_spec_group = base.ArgumentGroup(help='Basic level specification.')
    basic_level_spec_group.AddArgument(basic_level_spec_arg)
    basic_level_spec_group.AddArgument(basic_level_combine_arg)
    custom_level_spec_arg = base.Argument('--custom-level-spec', help=custom_level_help_text, type=ParseCustomLevel(api_version))
    custom_level_spec_group = base.ArgumentGroup(help='Custom level specification.')
    custom_level_spec_group.AddArgument(custom_level_spec_arg)
    level_spec_group = base.ArgumentGroup(help='Level specification.', mutex=True)
    level_spec_group.AddArgument(basic_level_spec_group)
    level_spec_group.AddArgument(custom_level_spec_group)
    level_spec_group.AddToParser(parser)