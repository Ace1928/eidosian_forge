from googlecloudsdk.api_lib.storage import insights_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage.insights.dataset_configs import log_util
from googlecloudsdk.command_lib.storage.insights.dataset_configs import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CreateLink(base.Command):
    """Create a link to a BigQuery instance."""
    detailed_help = {'DESCRIPTION': '\n      Create link to the customer BigQuery instance for Insights dataset config.\n      ', 'EXAMPLES': '\n\n      To create a link to the customer BigQuery instance for config name:\n      "my-config" in location "us-central1":\n\n          $ {command} my-config --location=us-central1\n\n      To create a link for the same dataset config with fully specified name:\n\n          $ {command} projects/foo/locations/us-central1/datasetConfigs/my-config\n      '}

    @staticmethod
    def Args(parser):
        resource_args.add_dataset_config_resource_arg(parser, 'to create link')

    def Run(self, args):
        client = insights_api.InsightsApi()
        dataset_config_relative_name = args.CONCEPTS.dataset_config.Parse().RelativeName()
        create_dataset_config_link_operation = client.create_dataset_config_link(dataset_config_relative_name)
        log_util.dataset_config_operation_started_and_status_log('Create link', dataset_config_relative_name, create_dataset_config_link_operation.name)