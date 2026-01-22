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
def _GetSubcommandTwoLevel(self, args):
    """Extract and return two level nested Airflow subcommand in unified shape.

    There are two ways to execute nested Airflow subcommands via gcloud, e.g.:
    1. {command} myenv users create -- -u User
    2. {command} myenv users -- create -u User
    The method returns here (users, create) in both cases.

    It is possible that first element of args.cmd_args will not be a nested
    subcommand, but that is ok as it will not break entire logic.
    So, essentially there can be subcommand_two_level = ['info', '--anonymize'].

    Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the .Args() method.

    Returns:
      subcommand_two_level: two level subcommand in unified format
    """
    subcommand_two_level = (args.subcommand, None)
    if args.subcommand_nested:
        subcommand_two_level = (args.subcommand, args.subcommand_nested)
    elif args.cmd_args:
        subcommand_two_level = (args.subcommand, args.cmd_args[0])
    return subcommand_two_level