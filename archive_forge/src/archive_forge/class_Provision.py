from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class Provision(base.DescribeCommand):
    """Provision an Apigee SaaS organization."""
    detailed_help = {'DESCRIPTION': '  {description}\n\n  `{command}` creates an Apigee organization and populates it with the\n  necessary child resources to be immediately useable.\n\n  This is a long running operation and could take anywhere from 10 minutes to 1\n  hour to complete.\n\n  At the moment, only trial organizations are supported.\n  ', 'EXAMPLES': '\n          To provision an organization for the active Cloud Platform project,\n          attached to a network named ``default\'\' run:\n\n              $ {command} --authorized-network=default\n\n          To provision an organization asynchronously and print only the name of\n          the launched operation, run:\n\n              $ {command} --authorized-network=default --async --format="value(name)"\n\n          To provision an organization for the project named ``my-proj\'\', with\n          analytics and runtimes located in ``us-central1\'\', run:\n\n              $ {command} --authorized-network=default --project=my-proj --analytics-region=us-central1 --runtime-location=us-central1-a\n          '}

    @staticmethod
    def Args(parser):
        parser.add_argument('--authorized-network', required=True, help='  Name of the network to which the provisioned organization should be attached.\n  This must be a VPC network peered through Service Networking. To get a list\n  of existing networks, run:\n\n      $ gcloud compute networks list\n\n  To check whether a network is peered through Service Networking, run:\n\n      $ gcloud services vpc-peerings list --network=NETWORK_NAME --service=servicenetworking.googleapis.com\n\n  To create a new network suitable for Apigee provisioning, choose a name for\n  the network and address range, and run:\n\n      $ gcloud compute networks create NETWORK_NAME --bgp-routing-mode=global --description=\'network for an Apigee trial\'\n      $ gcloud compute addresses create ADDRESS_RANGE_NAME --global --prefix-length=16 --description="peering range for an Apigee trial" --network=NETWORK_NAME --purpose=vpc_peering\n      $ gcloud services vpc-peerings connect --service=servicenetworking.googleapis.com --network=NETWORK_NAME --ranges=ADDRESS_RANGE_NAME')
        parser.add_argument('--analytics-region', help="Primary Cloud Platform region for analytics data storage. For valid values, see: https://docs.apigee.com/hybrid/latest/precog-provision.\n\nIf unspecified, the default is ``us-west1''")
        parser.add_argument('--runtime-location', help="Cloud Platform location for the runtime instance. For trial organizations, this is a compute zone. To get a list of valid zones, run `gcloud compute zones list`. If unspecified, the default is ``us-west1-a''.")
        parser.add_argument('--async', action='store_true', dest='async_', help="If set, returns immediately and outputs a description of the long running operation that was launched. Else, `{command}` will block until the organization has been provisioned.\n\nTo monitor the operation once it's been launched, run `{grandparent_command} operations describe OPERATION_NAME`.")

    def Run(self, args):
        """Run the provision command."""
        org_info = {'authorizedNetwork': args.authorized_network}
        if args.analytics_region:
            org_info['analyticsRegion'] = args.analytics_region
        if args.runtime_location:
            org_info['runtimeLocation'] = args.runtime_location
        project = properties.VALUES.core.project.Get()
        if project is None:
            exceptions.RequiredArgumentException('--project', 'Must provide a GCP project in which to provision the organization.')
        operation = apigee.ProjectsClient.ProvisionOrganization(project, org_info)
        apigee.OperationsClient.SplitName(operation)
        if args.async_:
            return operation
        log.info('Started provisioning operation %s', operation['name'])
        return waiter.WaitFor(apigee.LROPoller(operation['organization']), operation['uuid'], 'Provisioning organization', max_wait_ms=60 * 60 * 1000)