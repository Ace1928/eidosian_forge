from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.meta.apis import flags
from googlecloudsdk.core import properties
class Call(base.Command):
    """Calls an API method with specific parameters."""

    @staticmethod
    def Args(parser):
        flags.API_VERSION_FLAG.AddToParser(parser)
        flags.COLLECTION_FLAG.AddToParser(parser)
        ENFORCE_COLLECTION_FLAG.AddToParser(parser)
        flags.RAW_FLAG.AddToParser(parser)
        parser.AddDynamicPositional('method', action=flags.MethodDynamicPositionalAction, help='The name of the API method to invoke.')

    def Run(self, args):
        properties.VALUES.core.enable_gri.Set(True)
        response = args.method.Call()
        return response