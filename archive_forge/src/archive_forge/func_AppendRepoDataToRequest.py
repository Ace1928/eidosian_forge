from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from concurrent import futures
import encodings.idna  # pylint: disable=unused-import
import json
import mimetypes
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from googlecloudsdk.api_lib import artifacts
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import remote_repo_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import upgrade_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import edit
from googlecloudsdk.core.util import parallel
import requests
def AppendRepoDataToRequest(repo_ref, repo_args, request):
    """Adds repository data to CreateRepositoryRequest."""
    repo_name = repo_ref.repositoriesId
    location = GetLocation(repo_args)
    messages = _GetMessagesForResource(repo_ref)
    docker_format = messages.Repository.FormatValueValuesEnum.DOCKER
    repo_format = messages.Repository.FormatValueValuesEnum(repo_args.repository_format.upper())
    if repo_name in _ALLOWED_GCR_REPO_LOCATION:
        ValidateGcrRepo(repo_name, repo_format, location, docker_format)
    elif not _IsValidRepoName(repo_ref.repositoriesId):
        raise ar_exceptions.InvalidInputValueError(_INVALID_REPO_NAME_ERROR)
    if remote_repo_util.IsRemoteRepoRequest(repo_args):
        request = remote_repo_util.AppendRemoteRepoConfigToRequest(messages, repo_args, request)
    request.repository.name = repo_ref.RelativeName()
    request.repositoryId = repo_ref.repositoriesId
    request.repository.format = repo_format
    return request