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
def _IsPromptingSubcommand(s):
    if s in prompting_subcommands:
        pass
    elif s[0] in prompting_subcommands:
        s = s[0]
    else:
        return False
    return prompting_subcommands[s] is None or image_versions_command_util.CompareVersions(airflow_version, prompting_subcommands[s]) >= 0