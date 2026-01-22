from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
from googlecloudsdk.command_lib.util.anthos import structured_messages as sm
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def DefaultStreamOutHandler(result_holder, capture_output=False):
    """Default processing for streaming stdout from subprocess."""

    def HandleStdOut(line):
        if line:
            line.rstrip()
            log.Print(line)
        if capture_output:
            if not result_holder.stdout:
                result_holder.stdout = []
            result_holder.stdout.append(line)
    return HandleStdOut