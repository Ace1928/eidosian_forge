from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.command_lib.util.args import map_util
import six
def AddBuildEnvVarsFlags(parser):
    """Add flags for managing build environment variables.

  Args:
    parser: The argument parser.
  """
    map_util.AddUpdateMapFlags(parser, 'build-env-vars', long_name='build environment variables', key_type=BuildEnvVarKeyType, value_type=BuildEnvVarValueType)