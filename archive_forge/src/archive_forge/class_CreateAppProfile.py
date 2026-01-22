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
@base.ReleaseTracks(base.ReleaseTrack.GA)
class CreateAppProfile(base.CreateCommand):
    """Create a new Bigtable app profile."""
    detailed_help = {'EXAMPLES': textwrap.dedent('          To create an app profile with a multi-cluster routing policy, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --route-any\n\n          To create an app profile with a single-cluster routing policy which\n          routes all requests to `my-cluster-id`, run:\n\n            $ {command} my-single-cluster-app-profile --instance=my-instance-id --route-to=my-cluster-id\n\n          To create an app profile with a friendly description, run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --route-any --description="Routes requests for my use case"\n\n          To create an app profile with a request priority of PRIORITY_MEDIUM,\n          run:\n\n            $ {command} my-app-profile-id --instance=my-instance-id --route-any --priority=PRIORITY_MEDIUM\n\n          ')}

    @staticmethod
    def Args(parser):
        arguments.AddAppProfileResourceArg(parser, 'to create')
        arguments.ArgAdder(parser).AddDescription('app profile', required=False).AddAppProfileRouting().AddIsolation().AddForce('create')

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
        return app_profiles.Create(app_profile_ref, cluster=args.route_to, description=args.description, multi_cluster=args.route_any, restrict_to=args.restrict_to, transactional_writes=args.transactional_writes, priority=args.priority, force=args.force)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      ConflictingArgumentsException,
      OneOfArgumentsRequiredException:
        See _CreateAppProfile(...)

    Returns:
      Created resource.
    """
        app_profile_ref = args.CONCEPTS.app_profile.Parse()
        try:
            result = self._CreateAppProfile(app_profile_ref, args)
        except HttpError as e:
            util.FormatErrorMessages(e)
        else:
            log.CreatedResource(app_profile_ref.Name(), kind='app profile')
            return result