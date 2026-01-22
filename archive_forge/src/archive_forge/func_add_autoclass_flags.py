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
def add_autoclass_flags(parser):
    """Adds flags required for modifying Autoclass feature."""
    autoclass_group = parser.add_group(category='AUTOCLASS')
    autoclass_group.add_argument('--enable-autoclass', action=arg_parsers.StoreTrueFalseAction, help='The Autoclass feature automatically selects the best storage class for objects based on access patterns.')
    autoclass_group.add_argument('--autoclass-terminal-storage-class', help='The storage class that objects in the bucket eventually transition to if they are not read for a certain length of time. Only valid if Autoclass is enabled.')