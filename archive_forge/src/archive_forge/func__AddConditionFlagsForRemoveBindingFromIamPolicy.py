from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def _AddConditionFlagsForRemoveBindingFromIamPolicy(parser, condition_completer=None):
    """Create flags for condition and add to parser."""
    condition_intro = 'The condition of the binding that you want to remove. When the condition is\nexplicitly specified as `None` (`--condition=None`), a binding without a\ncondition is removed. Otherwise, only a binding with a condition that exactly\nmatches the specified condition (including the optional description) is removed.\nFor more on conditions, refer to the conditions overview guide:\nhttps://cloud.google.com/iam/docs/conditions-overview'
    help_str_condition = _ConditionHelpText(condition_intro)
    help_str_condition_from_file = '\nPath to a local JSON or YAML file that defines the condition.\nTo see available fields, see the help for `--condition`.'
    help_str_condition_all = '\nRemove all bindings with this role and principal, irrespective of any\nconditions.'
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument('--condition', type=_ConditionArgDict(), metavar='KEY=VALUE', completer=condition_completer, help=help_str_condition)
    condition_group.add_argument('--condition-from-file', type=arg_parsers.FileContents(), help=help_str_condition_from_file)
    condition_group.add_argument('--all', action='store_true', help=help_str_condition_all)