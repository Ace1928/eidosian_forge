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
def BypassConfirmationPrompt(self, args, airflow_version):
    """Bypasses confirmations with "yes" responses.

    Prevents certain Airflow CLI subcommands from presenting a confirmation
    prompting (which can make the gcloud CLI stop responding). When necessary,
    bypass confirmations with a "yes" response.

    Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the .Args() method.
      airflow_version: String, an Airflow semantic version.
    """
    prompting_subcommands = {'backfill': '1.10.6', 'delete_dag': None, ('dags', 'backfill'): None, ('dags', 'delete'): None, ('tasks', 'clear'): None, ('db', 'clean'): None}
    subcommand_two_level = self._GetSubcommandTwoLevel(args)

    def _IsPromptingSubcommand(s):
        if s in prompting_subcommands:
            pass
        elif s[0] in prompting_subcommands:
            s = s[0]
        else:
            return False
        return prompting_subcommands[s] is None or image_versions_command_util.CompareVersions(airflow_version, prompting_subcommands[s]) >= 0
    if _IsPromptingSubcommand(subcommand_two_level) and set(args.cmd_args or []).isdisjoint({'-y', '--yes'}):
        args.cmd_args = args.cmd_args or []
        args.cmd_args.append('--yes')