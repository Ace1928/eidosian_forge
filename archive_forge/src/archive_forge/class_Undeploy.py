from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.apigee import defaults
from googlecloudsdk.command_lib.apigee import resource_args
class Undeploy(base.SilentCommand):
    """Undeploy an Apigee API proxy from an environment."""
    detailed_help = {'EXAMPLES': "  To undeploy an API proxy called ``my-api'' from the ``test'' environment of\n  the active Cloud Platform project, run:\n\n    $ {command} --environment=test --api=my-api\n\n  To undeploy revision 3 of an `my-api` from the `test` environment of the\n  organization named ``test-org'', run:\n\n    $ {command} 3 --organization=test-org --environment=test --api=my-api\n  "}

    @staticmethod
    def Args(parser):
        help_text = {'api': 'API proxy to be undeployed. Must currently be deployed. To get a list of available deployed proxies, run `{{grandparent_command}} deployments list --environment=ENV`.', 'environment': 'Environment from which to undeploy the API proxy. To get a list of available environments, run `{{grandparent_command}} environments list`.', 'organization': 'Apigee organization of the proxy and environment.'}
        fallthroughs = [defaults.GCPProductOrganizationFallthrough(), defaults.StaticFallthrough('revision', 'auto')]
        resource_args.AddSingleResourceArgument(parser, 'organization.environment.api.revision', "API proxy revision to be undeployed and environment from which it should be removed.\n\nRevisions can either be a positive revision number, or the special value ``auto'', which will undeploy whatever revision is currently deployed. If revision is unspecified, the default is ``auto''.", fallthroughs=fallthroughs, help_texts=help_text)

    def Run(self, args):
        """Run the undeploy command."""
        identifiers = args.CONCEPTS.revision.Parse().AsDict()
        if identifiers['revisionsId'] == 'auto':
            del identifiers['revisionsId']
            defaults.FallBackToDeployedProxyRevision(identifiers)
        return apigee.APIsClient.Undeploy(identifiers)