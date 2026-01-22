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
def CheckSubcommandAirflowSupport(self, args, airflow_version):

    def _CheckIsSupportedSubcommand(command, airflow_version, from_version, to_version):
        if not image_versions_command_util.IsVersionInRange(airflow_version, from_version, to_version):
            _RaiseLackOfSupportError(command, airflow_version)

    def _RaiseLackOfSupportError(command, airflow_version):
        raise command_util.Error('The subcommand "{}" is not supported for Composer environments with Airflow version {}.'.format(command, airflow_version))
    subcommand, subcommand_nested = self._GetSubcommandTwoLevel(args)
    _CheckIsSupportedSubcommand(subcommand, airflow_version, self.SUBCOMMAND_ALLOWLIST[args.subcommand].from_version, self.SUBCOMMAND_ALLOWLIST[args.subcommand].to_version)
    if not self.SUBCOMMAND_ALLOWLIST[args.subcommand].allowed_nested_subcommands:
        return
    two_level_subcommand_string = '{} {}'.format(subcommand, subcommand_nested)
    if subcommand_nested in self.SUBCOMMAND_ALLOWLIST[args.subcommand].allowed_nested_subcommands:
        _CheckIsSupportedSubcommand(two_level_subcommand_string, airflow_version, self.SUBCOMMAND_ALLOWLIST[args.subcommand].allowed_nested_subcommands[subcommand_nested].from_version, self.SUBCOMMAND_ALLOWLIST[args.subcommand].allowed_nested_subcommands[subcommand_nested].to_version)
    else:
        _RaiseLackOfSupportError(two_level_subcommand_string, airflow_version)