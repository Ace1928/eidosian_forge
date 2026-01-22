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
def _GetLocationAndRepoPath(args, repo_format):
    """Get resource values and validate user input."""
    repo = _GetRequiredRepoValue(args)
    project = _GetRequiredProjectValue(args)
    location = _GetRequiredLocationValue(args)
    repo_path = project + '/' + repo
    repo = ar_requests.GetRepository('projects/{}/locations/{}/repositories/{}'.format(project, location, repo))
    if repo.format != repo_format:
        raise ar_exceptions.InvalidInputValueError('Invalid repository type {}. Valid type is {}.'.format(repo.format, repo_format))
    return (location, repo_path)