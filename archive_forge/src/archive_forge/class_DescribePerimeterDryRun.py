from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import perimeters
from googlecloudsdk.command_lib.accesscontextmanager import policies
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class DescribePerimeterDryRun(base.DescribeCommand):
    """Displays the dry-run mode configuration for a Service Perimeter."""
    _API_VERSION = 'v1'

    @staticmethod
    def Args(parser):
        perimeters.AddResourceArg(parser, 'to describe')

    def Run(self, args):
        client = zones_api.Client(version=self._API_VERSION)
        perimeter_ref = args.CONCEPTS.perimeter.Parse()
        policies.ValidateAccessPolicyArg(perimeter_ref, args)
        perimeter = client.Get(perimeter_ref)
        perimeters.GenerateDryRunConfigDiff(perimeter, self._API_VERSION)