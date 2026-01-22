from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddTriggerConfigFilePathArg(trigger_config):
    """Allow trigger config to be specified on the command line or STDIN.

  Args:
    trigger_config: the config of which the file path can be specified.
  """
    trigger_config.add_argument('--trigger-config', help='Path to Build Trigger config file (JSON or YAML format). For more details, see\nhttps://cloud.google.com/cloud-build/docs/api/reference/rest/v1/projects.triggers#BuildTrigger\n', metavar='PATH')