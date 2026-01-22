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
def _AddConditionFlagsForAddBindingToIamPolicy(parser):
    """Create flags for condition and add to parser."""
    condition_intro = 'A condition to include in the binding. When the condition is explicitly\nspecified as `None` (`--condition=None`), a binding without a condition is\nadded. When the condition is specified and is not `None`, `--role` cannot be a\nbasic role. Basic roles are `roles/editor`, `roles/owner`, and `roles/viewer`.\nFor more on conditions, refer to the conditions overview guide:\nhttps://cloud.google.com/iam/docs/conditions-overview'
    help_str_condition = _ConditionHelpText(condition_intro)
    help_str_condition_from_file = '\nPath to a local JSON or YAML file that defines the condition.\nTo see available fields, see the help for `--condition`.'
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument('--condition', type=_ConditionArgDict(), metavar='KEY=VALUE', help=help_str_condition)
    condition_group.add_argument('--condition-from-file', type=arg_parsers.FileContents(), help=help_str_condition_from_file)