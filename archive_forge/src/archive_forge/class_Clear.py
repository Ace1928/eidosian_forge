from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudiot import registries
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iot import resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
class Clear(base.Command):
    """Delete all credentials from a registry."""
    detailed_help = {'EXAMPLES': "          To delete all credentials from a registry in region 'us-central1', run:\n\n            $ {command} --region=us-central1 --registry=my-registry\n          "}

    @staticmethod
    def Args(parser):
        resource_args.AddRegistryResourceArg(parser, 'for which to clear credentials', positional=False)

    def Run(self, args):
        client = registries.RegistriesClient()
        registry_ref = args.CONCEPTS.registry.Parse()
        console_io.PromptContinue(message='This will delete ALL CREDENTIALS for registry [{}]'.format(registry_ref.Name()), cancel_on_no=True)
        response = client.Patch(registry_ref, credentials=[])
        log.status.Print('Cleared all credentials for registry [{}].'.format(registry_ref.Name()))
        return response