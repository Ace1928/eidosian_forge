from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.util.apis import arg_utils
def _OsPackageRemoteRepoHelpText(facade: str, hide_custom_remotes: bool) -> str:
    if hide_custom_remotes:
        return '({facade} only) Repository base for {facade_lower} remote repository.\nREMOTE_{facade_upper}_REPO must be one of: [{enums}].\n'.format(facade=facade, facade_lower=facade.lower(), facade_upper=facade.upper(), enums=_EnumsStrForFacade(facade))
    return '({facade} only) Repository base for {facade_lower} remote repository.\nREMOTE_{facade_upper}_REPO can be either:\n  - one of the following enums: [{enums}].\n  - an http/https custom registry uri (ex: https://my.{facade_lower}.registry)\n'.format(facade=facade, facade_lower=facade.lower(), facade_upper=facade.upper(), enums=_EnumsStrForFacade(facade))