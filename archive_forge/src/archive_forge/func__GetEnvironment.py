from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _GetEnvironment(args):
    """Prompt for environment if not provided.

  Environment is decided in the following order:
  - environment argument;
  - kuberun/environment gcloud config;
  - prompt user.

  Args:
    args: Environment, The args environment.

  Returns:
    A str representing the environment name.

  Raises:
    A ValueError if no environment is specified.
  """
    env = None
    if getattr(args, 'environment', None):
        env = args.environment
    elif properties.VALUES.kuberun.environment.IsExplicitlySet():
        env = properties.VALUES.kuberun.environment.Get()
    elif console_io.CanPrompt():
        env = console_io.PromptWithDefault('Environment name', default=None)
        log.status.Print('To make this the default environment, run `gcloud config set kuberun/environment {}`.\n'.format(env))
    if env:
        return env
    raise ValueError('Please specify an ENVIRONMENT to use this command.')