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
def AddComputeServiceAccountArg(parser, operation, roles):
    """Adds Compute service account arg."""
    help_text_pattern = "        A temporary virtual machine instance is created in your project during\n        {operation}.  {operation_capitalized} tooling on this temporary instance\n        must be authenticated.\n\n        A Compute Engine service account is an identity attached to an instance.\n        Its access tokens can be accessed through the instance metadata server\n        and can be used to authenticate {operation} tooling on the instance.\n\n        To set this option,  specify the email address corresponding to the\n        required Compute Engine service account. If not provided, the\n        {operation} on the temporary instance uses the project's default Compute\n        Engine service account.\n\n        At a minimum, you need to grant the following roles to the\n        specified Cloud Build service account:\n        "
    help_text_pattern += '\n'
    for role in roles:
        help_text_pattern += '        * ' + role + '\n'
    parser.add_argument('--compute-service-account', help=help_text_pattern.format(operation=operation, operation_capitalized=operation.capitalize()))