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
def GetYumSettingsSnippet(args):
    """Forms a Yum settings snippet to add to the yum.repos.d directory.

  Args:
    args: an argparse namespace. All the arguments that were provided to this
      command invocation.

  Returns:
    A yum settings snippet.
  """
    messages = ar_requests.GetMessages()
    location, repo_path = _GetLocationAndRepoPath(args, messages.Repository.FormatValueValuesEnum.YUM)
    repo = _GetRequiredRepoValue(args)
    project = _GetRequiredProjectValue(args)
    data = {'location': location, 'repo': repo, 'repo_path': repo_path}
    if IsPublicRepo(project, location, repo):
        yum_setting_template = yum.PUBLIC_TEMPLATE
    else:
        yum_setting_template = yum.DEFAULT_TEMPLATE
    return yum_setting_template.format(**data)