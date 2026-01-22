from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.auth import service_account
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util as ar_util
from googlecloudsdk.command_lib.artifacts.print_settings import apt
from googlecloudsdk.command_lib.artifacts.print_settings import gradle
from googlecloudsdk.command_lib.artifacts.print_settings import mvn
from googlecloudsdk.command_lib.artifacts.print_settings import npm
from googlecloudsdk.command_lib.artifacts.print_settings import python
from googlecloudsdk.command_lib.artifacts.print_settings import yum
from googlecloudsdk.core import config
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def GetGradleSnippet(args):
    """Forms a gradle snippet to add to the build.gradle file.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Returns:
    str, a gradle snippet to add to build.gradle.
  """
    messages = ar_requests.GetMessages()
    location, repo_path, maven_cfg = _GetLocationRepoPathAndMavenConfig(args, messages.Repository.FormatValueValuesEnum.MAVEN)
    sa_creds = _GetServiceAccountCreds(args)
    gradle_template = GetGradleTemplate(messages, maven_cfg, sa_creds)
    data = {'location': location, 'repo_path': repo_path}
    if sa_creds:
        data['username'] = '_json_key_base64'
        data['password'] = sa_creds
    else:
        data['extension_version'] = _EXT_VERSION
    return gradle_template.format(**data)