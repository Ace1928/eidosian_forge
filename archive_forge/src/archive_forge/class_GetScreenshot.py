from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
@base.UniverseCompatible
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class GetScreenshot(base.Command):
    """Capture a screenshot (JPEG image) of the virtual machine instance's display."""
    detailed_help = _DETAILED_HELP
    _display_output = False

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser, operation_type='get a screenshot from')
        parser.add_argument('--destination', help='Filename, including the path, to save the screenshot (JPEG image).')

    def _GetInstanceRef(self, holder, args):
        return flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=flags.GetInstanceZoneScopeLister(holder.client))

    def _GetInstance(self, holder, instance_ref):
        request = holder.client.messages.ComputeInstancesGetRequest(**instance_ref.AsDict())
        return holder.client.MakeRequests([(holder.client.apitools_client.instances, 'Get', request)])[0]

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        instance_ref = self._GetInstanceRef(holder, args)
        request = holder.client.messages.ComputeInstancesGetScreenshotRequest(**instance_ref.AsDict())
        response = holder.client.MakeRequests([(holder.client.apitools_client.instances, 'GetScreenshot', request)])[0]
        self._display_file_output = False
        if args.IsSpecified('destination'):
            with files.BinaryFileWriter(args.destination) as output:
                output.write(base64.b64decode(response.contents))
            self._resource_name = instance_ref.instance
            self._destination = args.destination
            self._display_file_output = True
        else:
            self._response_contents = response.contents
        return

    def Epilog(self, resources_were_displayed=False):
        if self._display_file_output:
            log.status.Print("Output screenshot for [{}] to '{}'.".format(self._resource_name, self._destination))
        else:
            sys.stdout.buffer.write(base64.b64decode(self._response_contents))