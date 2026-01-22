from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.domains import resource_args
from googlecloudsdk.command_lib.domains import util
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListGa(List):
    """List Cloud Domains registrations.

  List Cloud Domains registrations in the project.

  ## EXAMPLES

  To list all registrations in the project, run:

    $ {command}
  """

    @staticmethod
    def Args(parser):
        List.ArgsPerVersion(registrations.GA_API_VERSION, parser)