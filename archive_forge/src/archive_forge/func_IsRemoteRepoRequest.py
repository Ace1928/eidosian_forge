from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.util.apis import arg_utils
def IsRemoteRepoRequest(repo_args) -> bool:
    """Returns whether or not the repo mode specifies a remote repository."""
    return hasattr(repo_args, 'mode') and arg_utils.ChoiceToEnumName(repo_args.mode) == 'REMOTE_REPOSITORY'