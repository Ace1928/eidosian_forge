from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.util.apis import arg_utils
def _ChoiceToRemoteEnum(facade: str, remote_input: str):
    """Converts the remote repo input to a PublicRepository Enum message or None."""
    enums = _EnumsMessageForFacade(facade)
    name = arg_utils.ChoiceToEnumName(remote_input)
    try:
        return enums.lookup_by_name(name)
    except KeyError:
        return None