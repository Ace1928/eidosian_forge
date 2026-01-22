import enum
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import revision
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as services_exceptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.functions.v2.deploy import env_vars_util
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import secrets_mapping
from googlecloudsdk.command_lib.run import volumes
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def GetAndValidatePlatform(args, release_track, product):
    """Returns the platform to run on and validates specified flags.

  A given command may support multiple platforms, but not every flag is
  supported by every platform. This method validates that all specified flags
  are supported by the specified platform.

  Args:
    args: Namespace, The args namespace.
    release_track: base.ReleaseTrack, calliope release track.
    product: Product, which product the command was executed for (e.g. Run or
      Events).

  Raises:
    ArgumentError if an unknown platform type is found.
  """
    platform = platforms.GetPlatform()
    if platform == platforms.PLATFORM_MANAGED:
        VerifyManagedFlags(args, release_track, product)
    elif platform == platforms.PLATFORM_GKE:
        VerifyGKEFlags(args, release_track, product)
    elif platform == platforms.PLATFORM_KUBERNETES:
        VerifyKubernetesFlags(args, release_track, product)
    if platform not in platforms.PLATFORMS:
        raise serverless_exceptions.ArgumentError('Invalid target platform specified: [{}].\nAvailable platforms:\n{}'.format(platform, '\n'.join(['- {}: {}'.format(k, v) for k, v in platforms.PLATFORMS.items()])))
    return platform