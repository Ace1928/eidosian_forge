from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.vmware import dnsbindpermission
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.vmware import flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Grant(base.Command):
    """Revokes a DNS Bind Permission."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        flags.AddProjectArgToParser(parser)
        base.ASYNC_FLAG.AddToParser(parser)
        base.ASYNC_FLAG.SetDefault(parser, True)
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--user', required=False, help='        The consumer provided user whose permission needs to be revoked on the intranet VPC corresponding to the consumer project. If this field is not provided then the service-account should be provided.\n        ')
        group.add_argument('--service-account', required=False, help='        The consumer provided service account whose permission needs to be revoked on the intranet VPC corresponding to the consumer project. If this field is not provided then the user should be provided.\n        ')

    def Run(self, args):
        project = args.CONCEPTS.project.Parse()
        client = dnsbindpermission.DNSBindPermissionClient()
        is_async = args.async_
        operation = client.Revoke(project, user=args.user, service_account=args.service_account)
        if is_async:
            log.UpdatedResource(operation.name, kind='DNS Bind Permission', is_async=True)
            return
        dns_bind_permission = '{project}/locations/global/dnsBindPermission'.format(project=project.RelativeName())
        client.WaitForOperation(operation_ref=client.GetOperationRef(operation), message='waiting for DNS Bind Permission [{}] to be revoked'.format(dns_bind_permission), has_result=False)
        resource = client.Get(project)
        log.UpdatedResource(dns_bind_permission, kind='DNS Bind Permission')
        return resource