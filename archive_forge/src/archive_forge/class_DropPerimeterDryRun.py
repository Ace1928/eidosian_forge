from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class DropPerimeterDryRun(base.UpdateCommand):
    """Resets the dry-run state of a Service Perimeter."""
    _API_VERSION = 'v1'

    @staticmethod
    def Args(parser):
        perimeters.AddResourceArg(parser, 'to reset')
        parser.add_argument('--async', action='store_true', help='Return immediately, without waiting for the operation in\n            progress to complete.')

    def Run(self, args):
        client = zones_api.Client(version=self._API_VERSION)
        perimeter_ref = args.CONCEPTS.perimeter.Parse()
        policies.ValidateAccessPolicyArg(perimeter_ref, args)
        return client.UnsetSpec(perimeter_ref, use_explicit_dry_run_spec=False)