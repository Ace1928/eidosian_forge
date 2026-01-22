from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseBackoffMode(messages, backoff):
    """Parses BackoffMode of the Automation resource."""
    if not backoff:
        return backoff
    if backoff in BACKOFF_CHOICES_SHORT:
        backoff = 'BACKOFF_MODE_' + backoff
    return arg_utils.ChoiceToEnum(backoff, messages.Retry.BackoffModeValueValuesEnum, valid_choices=BACKOFF_CHOICES)