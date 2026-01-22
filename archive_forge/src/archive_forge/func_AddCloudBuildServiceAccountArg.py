from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
import time
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpError
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.services import enable_api as services_api
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as http_exc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.command_lib.compute.sole_tenancy import util as sole_tenancy_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding as encoding_util
import six
def AddCloudBuildServiceAccountArg(parser, operation, roles):
    """Adds Cloud Build service account arg."""
    help_text_pattern = "        Image import and export tools use Cloud Build to import and export images\n        to and from your project.\n        Cloud Build uses a specific service account to execute builds on your\n        behalf.\n        The Cloud Build service account generates an access token for other service\n        accounts and it is also used for authentication when building the artifacts\n        for the image import tool.\n\n        Use this flag to to specify a user-managed service account for\n        image import and export. If you don't specify this flag, Cloud Build\n        runs using your project's default Cloud Build service account.\n        To set this option, specify the email address of the desired\n        user-managed service account.\n        Note: You must specify the `--logs-location` flag when\n        you set a user-managed service account.\n\n        At minimum, the specified user-managed service account needs to have\n        the following roles assigned:\n        "
    help_text_pattern += '\n'
    for role in roles:
        help_text_pattern += '        * ' + role + '\n'
    parser.add_argument('--cloudbuild-service-account', help=help_text_pattern.format(operation=operation, operation_capitalized=operation.capitalize()))