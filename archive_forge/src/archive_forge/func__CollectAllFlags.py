from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def _CollectAllFlags(command):
    for flag in command.flags.values():
        all_flags[_FlagIndexKey(flag)] = _FlagIndex(flag)
    for subcommand in command.commands.values():
        _CollectAllFlags(subcommand)