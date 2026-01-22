from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.assured import endpoint_util
from googlecloudsdk.api_lib.assured import message_util
from googlecloudsdk.api_lib.assured import workloads as apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
class CreateWorkload(base.CreateCommand):
    """Create a new Assured Workloads environment."""
    detailed_help = _DETAILED_HELP

    def Run(self, args):
        """Run the create command."""
        with endpoint_util.AssuredWorkloadsEndpointOverridesFromRegion(release_track=self.ReleaseTrack(), region=args.location):
            parent = message_util.CreateAssuredParent(organization_id=args.organization, location=args.location)
            workload = message_util.CreateAssuredWorkload(display_name=args.display_name, compliance_regime=args.compliance_regime, partner=args.partner, partner_permissions=args.partner_permissions, billing_account=args.billing_account, next_rotation_time=args.next_rotation_time, rotation_period=args.rotation_period, labels=args.labels, provisioned_resources_parent=args.provisioned_resources_parent, resource_settings=args.resource_settings, enable_sovereign_controls=args.enable_sovereign_controls, release_track=self.ReleaseTrack())
            client = apis.WorkloadsClient(release_track=self.ReleaseTrack())
            self.created_resource = client.Create(external_id=args.external_identifier, parent=parent, workload=workload)
            return self.created_resource

    def Epilog(self, resources_were_displayed):
        log.CreatedResource(self.created_resource.name, kind='Assured Workloads environment')