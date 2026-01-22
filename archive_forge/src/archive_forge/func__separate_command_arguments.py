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
def _separate_command_arguments(command_string):
    """Move all flag arguments to back."""
    command_arguments = shlex.split(command_string)
    flag_args = [arg for arg in command_arguments if arg.startswith('--')]
    command_args = [arg for arg in command_arguments if not arg.startswith('--')]
    return command_args + flag_args