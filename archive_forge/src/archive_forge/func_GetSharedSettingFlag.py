from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.compute.reservations import flags as reservation_flags
def GetSharedSettingFlag(custom_name=None):
    """Gets the --share-setting flag."""
    help_text = '  Specify if this future reservation is shared, and if so, the type of sharing.\n  If you omit this flag, this value is local by default.\n  '
    return base.Argument(custom_name if custom_name else '--share-setting', choices=['local', 'projects'], help=help_text)