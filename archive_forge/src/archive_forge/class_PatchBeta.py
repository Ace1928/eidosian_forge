from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.sql import api_util as common_api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import instances as api_util
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.command_lib.sql import instances as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class PatchBeta(base.UpdateCommand):
    """Updates the settings of a Cloud SQL instance."""

    def Run(self, args):
        return RunBasePatchCommand(args, self.ReleaseTrack())

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command."""
        AddBaseArgs(parser)
        flags.AddZone(parser, help_text='Preferred Compute Engine zone (e.g. us-central1-a, us-central1-b, etc.). WARNING: Instance may be restarted.')
        AddBetaArgs(parser)
        flags.AddDatabaseVersion(parser, restrict_choices=False, support_default_version=False)