from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_per_object_retention_flags(parser, is_update=False):
    """Adds the flags for object retention lock.

  Args:
    parser (parser_arguments.ArgumentInterceptor): Parser passed to surface.
    is_update (bool): True if flags are for the objects update command.
  """
    retention_group = parser.add_group(category='RETENTION')
    if is_update:
        subject = 'object'
        retention_group.add_argument('--clear-retention', action='store_true', help='Clears object retention settings and unlocks the configuration. Requires --override-unlocked-retention flag as confirmation.')
        retention_group.add_argument('--override-unlocked-retention', action='store_true', help='Needed for certain retention configuration modifications, such as clearing retention settings and reducing retention time. Note that locked configurations cannot be edited even with this flag.')
        override_note = ' Requires --override-unlocked-retention flag to shorten the retain-until time in unlocked configurations.'
    else:
        subject = 'destination object'
        override_note = ''
    retention_group.add_argument('--retention-mode', choices=sorted([option.value for option in RetentionMode]), help='Sets the {} retention mode to either "Locked" or "Unlocked". When retention mode is "Locked", the retain until time can only be increased.'.format(subject))
    retention_group.add_argument('--retain-until', type=arg_parsers.Datetime.Parse, help='Ensures the {} is retained until the specified time in RFC 3339 format.'.format(subject) + override_note, metavar='DATETIME')