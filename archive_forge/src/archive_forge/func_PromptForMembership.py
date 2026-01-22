from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def PromptForMembership(memberships=None, flag='--membership', message='Please specify a membership:\n', cancel=True):
    """Prompt the user for a membership from a list of memberships in the fleet.

  This method is referenced by fleet and feature commands as a fallthrough
  for getting the memberships when required.

  Args:
    memberships: List of memberships to prompt from
    flag: The name of the membership flag, used in the error message
    message: The message given to the user describing the prompt.
    cancel: Whether to include a "cancel" option.

  Returns:
    The membership specified by the user (str), or None if unable to prompt.

  Raises:
    OperationCancelledError if the prompt is cancelled by user
    RequiredArgumentException if the console is unable to prompt
  """
    if not console_io.CanPrompt():
        raise calliope_exceptions.RequiredArgumentException(flag, 'Cannot prompt a console for membership. Membership is required. Please specify `{}` to select at least one membership.'.format(flag))
    if memberships is None:
        memberships, unreachable = api_util.ListMembershipsFull()
        if unreachable:
            raise exceptions.Error('Locations {} are currently unreachable. Please specify memberships using `--location` or the full resource name (projects/*/locations/*/memberships/*)'.format(unreachable))
    if not memberships:
        raise exceptions.Error('No Memberships available in the fleet.')
    idx = console_io.PromptChoice(MembershipPartialNames(memberships), message=message, cancel_option=cancel)
    return memberships[idx] if idx is not None else None