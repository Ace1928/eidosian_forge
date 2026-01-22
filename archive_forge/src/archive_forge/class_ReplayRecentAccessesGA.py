from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding_helper
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.iam import assist
from googlecloudsdk.api_lib.iam.simulator import operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class ReplayRecentAccessesGA(base.Command):
    """Determine affected recent access attempts before IAM policy change deployment."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        parser.add_argument('resource', metavar='RESOURCE', help='\n        Full resource name to simulate the IAM policy for.\n\n        See: https://cloud.google.com/apis/design/resource_names#full_resource_name.\n        ')
        parser.add_argument('policy_file', metavar='POLICY_FILE', help='\n        Path to a local JSON or YAML formatted file containing a valid policy.\n\n        The output of the `get-iam-policy` command is a valid file, as is any\n        JSON or YAML file conforming to the structure of a Policy. See\n        [the Policy reference](https://cloud.google.com/iam/reference/rest/v1/Policy)\n        for details.\n        ')

    def Run(self, args):
        api_version = 'v1'
        client, messages = assist.GetSimulatorClientAndMessages(api_version)
        policy = iam_util.ParsePolicyFile(args.policy_file, messages.GoogleIamV1Policy)
        policy.version = iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION
        additional_property = messages.GoogleCloudPolicysimulatorV1ReplayConfig.PolicyOverlayValue.AdditionalProperty(key=args.resource, value=policy)
        create_replay_parent = 'projects/{0}/locations/global'.format(properties.VALUES.core.project.Get(required=True))
        overlay = messages.GoogleCloudPolicysimulatorV1ReplayConfig.PolicyOverlayValue(additionalProperties=[additional_property])
        config = messages.GoogleCloudPolicysimulatorV1ReplayConfig(policyOverlay=overlay)
        replay = messages.GoogleCloudPolicysimulatorV1Replay(config=config)
        create_replay_request = messages.PolicysimulatorProjectsLocationsReplaysCreateRequest(googleCloudPolicysimulatorV1Replay=replay, parent=create_replay_parent)
        create_replay_response = client.ProjectsLocationsReplaysService.Create(client.ProjectsLocationsReplaysService(client), create_replay_request)
        operations_client = operations.Client.FromApiVersion(api_version)
        operation_response_raw = operations_client.WaitForOperation(create_replay_response, 'Waiting for operation [{}] to complete'.format(create_replay_response.name))
        operation_response = encoding_helper.JsonToMessage(messages.GoogleCloudPolicysimulatorV1Replay, encoding_helper.MessageToJson(operation_response_raw))
        if not operation_response.resultsSummary or not operation_response.resultsSummary.differenceCount:
            log.err.Print('No access changes found in the replay.\n')
        list_replay_result_request = messages.PolicysimulatorProjectsLocationsReplaysResultsListRequest(parent=operation_response.name)
        replay_result_service = client.ProjectsLocationsReplaysResultsService(client)
        return list_pager.YieldFromList(replay_result_service, list_replay_result_request, batch_size=1000, field='replayResults', batch_size_attribute='pageSize')