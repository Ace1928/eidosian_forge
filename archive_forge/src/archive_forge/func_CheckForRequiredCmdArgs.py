from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import random
import re
import time
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def CheckForRequiredCmdArgs(self, args):
    """Prevents running Airflow CLI commands without required arguments.

    Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the .Args() method.
    """
    required_cmd_args = {('users', 'create'): [['-p', '--password', '--use-random-password']]}

    def _StringifyRequiredCmdArgs(cmd_args):
        quoted_args = ['"{}"'.format(a) for a in cmd_args]
        return '[{}]'.format(', '.join(quoted_args))
    subcommand_two_level = self._GetSubcommandTwoLevel(args)
    for subcommand_required_cmd_args in required_cmd_args.get(subcommand_two_level, []):
        if set(subcommand_required_cmd_args).isdisjoint(set(args.cmd_args or [])):
            raise command_util.Error('The subcommand "{}" requires one of the following command line arguments: {}.'.format(' '.join(subcommand_two_level), _StringifyRequiredCmdArgs(subcommand_required_cmd_args)))