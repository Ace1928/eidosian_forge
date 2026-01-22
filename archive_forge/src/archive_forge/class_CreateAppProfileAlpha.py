from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.bigtable import app_profiles
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CreateAppProfileAlpha(CreateAppProfileBeta):
    """Create a new Bigtable app profile."""

    @staticmethod
    def Args(parser):
        arguments.AddAppProfileResourceArg(parser, 'to create')
        arguments.ArgAdder(parser).AddDescription('app profile', required=False).AddAppProfileRouting(allow_failover_radius=True, allow_row_affinity=True).AddIsolation(allow_data_boost=True).AddForce('create')

    def _CreateAppProfile(self, app_profile_ref, args):
        """Creates an AppProfile with the given arguments.

    Args:
      app_profile_ref: A resource reference of the new app profile.
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      ConflictingArgumentsException,
      OneOfArgumentsRequiredException:
        See app_profiles.Create(...)

    Returns:
      Created app profile resource object.
    """
        return app_profiles.Create(app_profile_ref, cluster=args.route_to, description=args.description, multi_cluster=args.route_any, restrict_to=args.restrict_to, failover_radius=args.failover_radius, transactional_writes=args.transactional_writes, row_affinity=args.row_affinity, priority=args.priority, data_boost=args.data_boost, data_boost_compute_billing_owner=args.data_boost_compute_billing_owner, force=args.force)