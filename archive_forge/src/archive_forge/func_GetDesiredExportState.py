from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
from six.moves.urllib.parse import urlparse
def GetDesiredExportState(args, psl):
    """Returns the export DesiredState enum from arguments."""
    if args.export_desired_state == 'paused':
        return psl.ExportConfig.DesiredStateValueValuesEnum.PAUSED
    return psl.ExportConfig.DesiredStateValueValuesEnum.ACTIVE