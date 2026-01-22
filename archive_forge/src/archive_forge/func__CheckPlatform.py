from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
def _CheckPlatform(self):
    if platforms.GetPlatform() != platforms.PLATFORM_MANAGED:
        raise exceptions.PlatformError('This command group only supports listing regions for Cloud Run (fully managed).')