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
def GetBaseSnoozeMessageFromArgs(args, snooze_class, update=False):
    """Returns the base snooze from args."""
    if args.IsSpecified('snooze_from_file'):
        snooze_string = args.snooze_from_file
        if update:
            snooze = MessageFromString(snooze_string, snooze_class, 'Snooze', field_deletions=SNOOZE_FIELD_DELETIONS)
        else:
            snooze = MessageFromString(snooze_string, snooze_class, 'Snooze')
    else:
        snooze = snooze_class()
    return snooze