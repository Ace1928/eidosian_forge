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
def _validate_command(self, command_string):
    """Validate a single command."""
    command_arguments = _separate_command_arguments(command_string)
    command_success, command_node, flag_arguments = self._validate_command_prefix(command_arguments, command_string)
    if not command_success:
        return
    flag_success = self._validate_command_suffix(command_node, flag_arguments, command_string)
    if not flag_success:
        return
    self._store_validation_results(True, command_string)