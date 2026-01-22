from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.functions import api_enablement
from googlecloudsdk.api_lib.functions import cmek_util
from googlecloudsdk.api_lib.functions import secrets as secrets_util
from googlecloudsdk.api_lib.functions.v1 import env_vars as env_vars_api_util
from googlecloudsdk.api_lib.functions.v1 import exceptions as function_exceptions
from googlecloudsdk.api_lib.functions.v1 import util as api_util
from googlecloudsdk.api_lib.functions.v2 import client as v2_client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.functions import flags
from googlecloudsdk.command_lib.functions import secrets_config
from googlecloudsdk.command_lib.functions.v1.deploy import enum_util
from googlecloudsdk.command_lib.functions.v1.deploy import labels_util
from googlecloudsdk.command_lib.functions.v1.deploy import source_util
from googlecloudsdk.command_lib.functions.v1.deploy import trigger_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from six.moves import urllib
def _ApplyDockerRegistryArgsToFunction(function, args):
    """Populates the `docker_registry` field of a Cloud Function message.

  Args:
    function: Cloud function message to be checked and populated.
    args: All CLI arguments.

  Returns:
    updated_fields: update mask containing the list of fields to be updated.

  Raises:
    InvalidArgumentException: If Container Registry is specified for a CMEK
    deployment (CMEK is only supported by Artifact Registry).
  """
    updated_fields = []
    if args.IsSpecified('docker_registry'):
        kms_key = function.kmsKeyName
        if args.IsSpecified('kms_key') or args.IsSpecified('clear_kms_key'):
            kms_key = None if args.clear_kms_key else args.kms_key
        if kms_key is not None and args.docker_registry == 'container-registry':
            raise calliope_exceptions.InvalidArgumentException('--docker-registry', 'CMEK deployments are not supported by Container Registry.Please either use Artifact Registry or do not specify a KMS key for the function (you may need to clear it).')
        function.dockerRegistry = enum_util.ParseDockerRegistry(args.docker_registry)
        updated_fields.append('dockerRegistry')
    return updated_fields