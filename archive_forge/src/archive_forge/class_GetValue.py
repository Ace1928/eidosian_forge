from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudiot import devices
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iot import resource_args
from googlecloudsdk.command_lib.iot import util
class GetValue(base.Command):
    """Show the binary data of a device's latest configuration."""
    detailed_help = {'EXAMPLES': "          To show the binary data of the latest configuration of a device in region 'us-central1', run:\n\n            $ {command} --region=us-central1 --registry=my-registry --device=my-device\n          "}

    @staticmethod
    def Args(parser):
        parser.display_info.AddFormat('get[terminator=""](.)')
        resource_args.AddDeviceResourceArg(parser, 'for the configuration to get the value of', positional=False)

    def Run(self, args):
        client = devices.DevicesClient()
        device_ref = args.CONCEPTS.device.Parse()
        device = client.Get(device_ref)
        try:
            data = device.config.binaryData
        except AttributeError:
            raise util.BadDeviceError('Device [{}] is missing configuration data.'.format(device_ref.Name()))
        return data