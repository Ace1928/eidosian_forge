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
def _PromptForConditionAddBindingToIamPolicy(policy):
    """Prompt user for a condition when adding binding."""
    prompt_message = 'The policy contains bindings with conditions, so specifying a condition is required when adding a binding. Please specify a condition.'
    conditions = PromptChoicesForAddBindingToIamPolicy(policy)
    condition_keys = [c[0] for c in conditions]
    condition_index = console_io.PromptChoice(condition_keys, prompt_string=prompt_message)
    if condition_index == len(conditions) - 1:
        return _PromptForNewCondition()
    return _ToDictCondition(conditions[condition_index][1])