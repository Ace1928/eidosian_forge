from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import copy
import json
import shlex
from googlecloudsdk import gcloud_main
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _read_commands_from_file(commands_file):
    with files.FileReader(commands_file) as f:
        command_file_data = json.load(f)
    command_strings = []
    for command_data in command_file_data:
        command_strings.append(command_data['command_string'])
    return command_strings