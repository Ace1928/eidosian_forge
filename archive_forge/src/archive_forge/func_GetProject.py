from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.error_reporting import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.error_reporting import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def GetProject(self, args):
    """Get project name."""
    return properties.VALUES.core.project.Get(required=True)