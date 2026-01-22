from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from typing import Iterator
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources as fleet_resources
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as fleet_messages
class RolloutFlags:
    """Add flags to the fleet rollout command surface."""

    def __init__(self, parser: parser_arguments.ArgumentInterceptor, release_track: base.ReleaseTrack=base.ReleaseTrack.ALPHA):
        self._parser = parser
        self._release_track = release_track

    @property
    def parser(self):
        return self._parser

    @property
    def release_track(self):
        return self._release_track

    def AddAsync(self):
        base.ASYNC_FLAG.AddToParser(self.parser)

    def AddDisplayName(self):
        self.parser.add_argument('--display-name', type=str, help=textwrap.dedent('            Display name of the rollout to be created (optional). 4-30\n            characters, alphanumeric and [ \'"!-] only.\n        '))

    def AddLabels(self):
        self.parser.add_argument('--labels', help='Labels for the rollout.', metavar='KEY=VALUE', type=arg_parsers.ArgDict())

    def AddManagedRolloutConfig(self):
        managed_rollout_config_group = self.parser.add_group(help='Configurations for the Rollout. Waves are assigned automatically.')
        self._AddSoakDuration(managed_rollout_config_group)

    def _AddSoakDuration(self, managed_rollout_config_group: parser_arguments.ArgumentInterceptor):
        managed_rollout_config_group.add_argument('--soak-duration', help=textwrap.dedent('          Soak time before starting the next wave. e.g. `4h`, `2d6h`.\n\n          See $ gcloud topic datetimes for information on duration formats.'), type=arg_parsers.Duration())

    def AddRolloutResourceArg(self):
        fleet_resources.AddRolloutResourceArg(parser=self.parser, api_version=util.VERSION_MAP[self.release_track])

    def AddFeatureUpdate(self):
        feature_update_mutex_group = self.parser.add_mutually_exclusive_group(help='Feature config to use for Rollout.')
        self._AddSecurityPostureConfig(feature_update_mutex_group)
        self._AddBinaryAuthorizationConfig(feature_update_mutex_group)

    def _AddSecurityPostureConfig(self, feature_update_mutex_group: parser_arguments.ArgumentInterceptor):
        security_posture_config_group = feature_update_mutex_group.add_group(help='Security posture config.')
        self._AddSecurityPostureMode(security_posture_config_group)
        self._AddWorkloadVulnerabilityScanningMode(security_posture_config_group)

    def _AddSecurityPostureMode(self, security_posture_config_group: parser_arguments.ArgumentInterceptor):
        security_posture_config_group.add_argument('--security-posture', choices=['disabled', 'standard'], default=None, help=textwrap.dedent('          To apply standard security posture to clusters in the fleet,\n\n            $ {command} --security-posture=standard\n\n          '))

    def _AddWorkloadVulnerabilityScanningMode(self, security_posture_config_group: parser_arguments.ArgumentInterceptor):
        security_posture_config_group.add_argument('--workload-vulnerability-scanning', choices=['disabled', 'standard', 'enterprise'], default=None, help=textwrap.dedent('            To apply standard vulnerability scanning to clusters in the fleet,\n\n              $ {command} --workload-vulnerability-scanning=standard\n\n            '))

    def _AddBinaryAuthorizationConfig(self, feature_update_mutex_group: parser_arguments.ArgumentInterceptor):
        binary_authorization_config_group = feature_update_mutex_group.add_group(help='Binary Authorization config.')
        self._AddBinauthzEvaluationMode(binary_authorization_config_group)
        self._AddBinauthzPolicyBindings(binary_authorization_config_group)

    def _AddBinauthzEvaluationMode(self, binary_authorization_config_group: parser_arguments.ArgumentInterceptor):
        binary_authorization_config_group.add_argument('--binauthz-evaluation-mode', choices=['disabled', 'policy-bindings'], type=lambda x: x.replace('_', '-').lower(), default=None, help=textwrap.dedent('          Configure binary authorization mode for clusters to onboard the fleet,\n\n            $ {command} --binauthz-evaluation-mode=policy-bindings\n\n          '))

    def _AddBinauthzPolicyBindings(self, binary_authorization_config_group: parser_arguments.ArgumentInterceptor):
        platform_policy_type = arg_parsers.RegexpValidator(_BINAUTHZ_GKE_POLICY_REGEX, 'GKE policy resource names have the following format: `projects/{project_number}/platforms/gke/policies/{policy_id}`')
        binary_authorization_config_group.add_argument('--binauthz-policy-bindings', default=None, action='append', metavar='name=BINAUTHZ_POLICY', help=textwrap.dedent('          The relative resource name of the Binary Authorization policy to audit\n          and/or enforce. GKE policies have the following format:\n          `projects/{project_number}/platforms/gke/policies/{policy_id}`.'), type=arg_parsers.ArgDict(spec={'name': platform_policy_type}, required_keys=['name'], max_length=1))