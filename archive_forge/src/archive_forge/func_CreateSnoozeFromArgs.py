from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def CreateSnoozeFromArgs(args, messages):
    """Builds a Snooze message from args."""
    snooze_base_flags = ['--display-name', '--snooze-from-file']
    ValidateAtleastOneSpecified(args, snooze_base_flags)
    snooze = GetBaseSnoozeMessageFromArgs(args, messages.Snooze)
    ModifySnooze(snooze, messages, display_name=args.display_name, criteria_policies=args.criteria_policies, start_time=args.start_time, end_time=args.end_time)
    return snooze