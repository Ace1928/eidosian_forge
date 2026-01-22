from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app import service_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
import six
import six.moves
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Migrate(base.Command):
    """Migrate traffic from one version to another for a set of services."""
    detailed_help = {'EXAMPLES': '          This only works for automatically scaled Standard versions.\n          To migrate from one version to another for all services where there\n          is a version v2 and shut down the previous version, run:\n\n            $ {command} v2\n\n          To migrate from one version to another for a specific service, run:\n\n            $ {command} v2 --service="s1"\n          '}

    @staticmethod
    def Args(parser):
        parser.add_argument('version', help='The version to migrate to.')
        parser.add_argument('--service', '-s', help='If specified, only migrate versions from the given service.')

    def Run(self, args):
        client = appengine_api_client.GetApiClientForTrack(self.ReleaseTrack())
        if args.service:
            service = client.GetServiceResource(args.service)
            traffic_split = {}
            if service.split:
                for split in service.split.allocations.additionalProperties:
                    traffic_split[split.key] = split.value
            services = [service_util.Service(client.project, service.id, traffic_split)]
        else:
            services = client.ListServices()
        all_versions = client.ListVersions(services)
        if args.version not in {v.id for v in all_versions}:
            if args.service:
                raise VersionsMigrateError('Version [{0}/{1}] does not exist.'.format(args.service, args.version))
            else:
                raise VersionsMigrateError('Version [{0}] does not exist.'.format(args.version))
        service_names = {v.service for v in all_versions if v.id == args.version}

        def WillBeMigrated(v):
            return v.service in service_names and v.traffic_split and (v.traffic_split > 0) and (v.id != args.version)
        versions_to_migrate = list(six.moves.filter(WillBeMigrated, all_versions))
        for version in versions_to_migrate:
            short_name = '{0}/{1}'.format(version.service, version.id)
            promoted_name = '{0}/{1}'.format(version.service, args.version)
            log.status.Print('Migrating all traffic from version [{0}] to [{1}]'.format(short_name, promoted_name))
        console_io.PromptContinue(cancel_on_no=True)
        errors = {}
        for service in sorted(set([v.service for v in versions_to_migrate])):
            allocations = {args.version: 1.0}
            try:
                operations_util.CallAndCollectOpErrors(client.SetTrafficSplit, service, allocations, shard_by='ip', migrate=True)
            except operations_util.MiscOperationError as err:
                errors[service] = six.text_type(err)
        if errors:
            error_string = 'Issues migrating all traffic of service(s): [{0}]\n\n{1}'.format(', '.join(list(errors.keys())), '\n\n'.join(list(errors.values())))
            raise VersionsMigrateError(error_string)