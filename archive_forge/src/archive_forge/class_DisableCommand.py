from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import exceptions as core_api_exceptions
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import base as hub_base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import retry
class DisableCommand(FeatureCommand, calliope_base.DeleteCommand):
    """Base class for the command that disables a Feature."""

    @staticmethod
    def Args(parser):
        parser.add_argument('--force', action='store_true', help='Disable this feature, even if it is currently in use. Force disablement may result in unexpected behavior.')

    def Run(self, args):
        return self.Disable(args.force)

    def Disable(self, force):
        try:
            op = self.hubclient.DeleteFeature(self.FeatureResourceName(), force=force)
        except apitools_exceptions.HttpNotFoundError:
            return
        message = 'Waiting for Feature {} to be deleted'.format(self.feature.display_name)
        self.WaitForHubOp(self.hubclient.resourceless_waiter, op, message=message, warnings=False)