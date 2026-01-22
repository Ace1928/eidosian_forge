import typing
from typing import Mapping, Tuple
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import yaml
def GetUserWorkloadsSecret(environment_ref: 'Resource', secret_name: str, release_track: base.ReleaseTrack=base.ReleaseTrack.ALPHA) -> typing.Union['composer_v1alpha2_messages.UserWorkloadsSecret', 'composer_v1beta1_messages.UserWorkloadsSecret', 'composer_v1_messages.UserWorkloadsSecret']:
    """Calls the Composer Environments.GetUserWorkloadsSecret method.

  Args:
    environment_ref: Resource, the Composer environment resource to get a user
      workloads Secret for.
    secret_name: string, name of the Kubernetes Secret.
    release_track: base.ReleaseTrack, the release track of the command. Will
      dictate which Composer client library will be used.

  Returns:
    UserWorkloadsSecret: user workloads Secret.
  """
    message_module = api_util.GetMessagesModule(release_track=release_track)
    user_workloads_secret_name = f'{environment_ref.RelativeName()}/userWorkloadsSecrets/{secret_name}'
    request_message = message_module.ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsGetRequest(name=user_workloads_secret_name)
    return GetService(release_track=release_track).Get(request_message)