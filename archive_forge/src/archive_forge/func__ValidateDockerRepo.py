from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.artifacts import containeranalysis_util as ca_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def _ValidateDockerRepo(repo_name):
    repo = ar_requests.GetRepository(repo_name)
    messages = ar_requests.GetMessages()
    if repo.format != messages.Repository.FormatValueValuesEnum.DOCKER:
        raise ar_exceptions.InvalidInputValueError('Invalid repository type {}. The `artifacts docker` command group can only be used on Docker repositories.'.format(repo.format))