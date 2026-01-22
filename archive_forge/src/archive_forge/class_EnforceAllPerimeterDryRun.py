from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.accesscontextmanager import zones as zones_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class EnforceAllPerimeterDryRun(base.UpdateCommand):
    """Enforces the dry-run mode configuration for all Service Perimeters."""
    _API_VERSION = 'v1'

    @staticmethod
    def Args(parser):
        parser.add_argument('--policy', metavar='policy', default=None, help='The parent Access Policy which owns all Service Perimeters in\n                scope for the commit operation.')
        parser.add_argument('--etag', metavar='etag', default=None, help='The etag for the version of the Access Policy that this\n                operation is to be performed on. If, at the time of the\n                operation, the etag for the Access Policy stored in Access\n                Context Manager is different from the specified etag, then the\n                commit operation will not be performed and the call will fail.\n                If etag is not provided, the operation will be performed as if a\n                valid etag is provided.')

    def Run(self, args):
        client = zones_api.Client(version=self._API_VERSION)
        policy_id = policies.GetDefaultPolicy()
        if args.IsSpecified('policy'):
            policy_id = args.policy
        policy_ref = resources.REGISTRY.Parse(policy_id, collection='accesscontextmanager.accessPolicies')
        return client.Commit(policy_ref, args.etag)