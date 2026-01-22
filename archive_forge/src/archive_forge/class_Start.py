from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.app import version_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
import six
class Start(base.Command):
    """Start serving specified versions.

  This command starts serving the specified versions. It may only be used if the
  scaling module for your service has been set to manual.
  """
    detailed_help = {'EXAMPLES': '          To start a specific version across all services, run:\n\n            $ {command} v1\n\n          To start multiple named versions across all services, run:\n\n            $ {command} v1 v2\n\n          To start a single version on a single service, run:\n\n            $ {command} --service=servicename v1\n\n          To start multiple versions in a single service, run:\n\n            $ {command} --service=servicename v1 v2\n          '}

    @staticmethod
    def Args(parser):
        parser.add_argument('versions', nargs='+', help='The versions to start. (optionally filtered by the --service flag).')
        parser.add_argument('--service', '-s', help='If specified, only start versions from the given service.')

    def Run(self, args):
        api_client = appengine_api_client.GetApiClientForTrack(self.ReleaseTrack())
        services = api_client.ListServices()
        versions = version_util.GetMatchingVersions(api_client.ListVersions(services), args.versions, args.service)
        if not versions:
            log.warning('No matching versions found.')
            return
        fmt = 'list[title="Starting the following versions:"]'
        resource_printer.Print(versions, fmt, out=log.status)
        console_io.PromptContinue(cancel_on_no=True)
        errors = {}
        for version in sorted(versions, key=str):
            try:
                with progress_tracker.ProgressTracker('Starting [{0}]'.format(version)):
                    operations_util.CallAndCollectOpErrors(api_client.StartVersion, version.service, version.id)
            except operations_util.MiscOperationError as err:
                errors[version] = six.text_type(err)
        if errors:
            printable_errors = {}
            for version, error_msg in errors.items():
                short_name = '[{0}/{1}]'.format(version.service, version.id)
                printable_errors[short_name] = '{0}: {1}'.format(short_name, error_msg)
            raise VersionsStartError('Issues starting version(s): {0}\n\n'.format(', '.join(list(printable_errors.keys()))) + '\n\n'.join(list(printable_errors.values())))