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
def AddBasicSpecArgs(parser, api_version):
    """Add args for basic spec (with no custom spec)."""
    basic_level_help_text = 'Path to a file containing a list of basic access level conditions.\n\nAn access level condition file is a YAML-formatted list of conditions, which are YAML objects representing a Condition as described in the API reference. For example:\n\n    ```\n     - ipSubnetworks:\n       - 162.222.181.197/24\n       - 2001:db8::/48\n     - members:\n       - user:user@example.com\n    ```'
    basic_level_spec_arg = base.Argument('--basic-level-spec', help=basic_level_help_text, type=ParseBasicLevelConditions(api_version))
    basic_level_combine_arg = GetCombineFunctionEnumMapper(api_version=api_version).choice_arg
    basic_level_spec_arg.AddToParser(parser)
    basic_level_combine_arg.AddToParser(parser)