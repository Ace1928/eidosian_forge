from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import app_profiles
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bigtable import arguments
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class ListAppProfilesBeta(ListAppProfilesGA):
    """List Bigtable app profiles."""

    @staticmethod
    def Args(parser):
        arguments.AddInstanceResourceArg(parser, 'to list app profiles for')
        parser.display_info.AddTransforms({'routingInfo': _TransformAppProfileToRoutingInfo, 'isolationMode': _TransformAppProfileToIsolationMode, 'standardIsolationPriority': _TransformAppProfileToStandardIsolationPriority, 'dataBoostComputeBillingOwner': _TransformAppProfileToDataBoostComputeBillingOwner})
        parser.display_info.AddFormat('\n          table(\n            name.basename():sort=1,\n            description:wrap,\n            routingInfo():wrap:label=ROUTING,\n            singleClusterRouting.allowTransactionalWrites.yesno("Yes"):label=TRANSACTIONAL_WRITES,\n            isolationMode():label=ISOLATION_MODE,\n            standardIsolationPriority():label=STANDARD_ISOLATION_PRIORITY,\n            dataBoostComputeBillingOwner():label=DATA_BOOST_COMPUTE_BILLING_OWNER\n          )\n        ')