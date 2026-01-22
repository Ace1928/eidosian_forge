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
def _CaptureStdOut(result_holder, output_message=None, resource_output=None, raw_output=None):
    """Update OperationResult from OutputMessage or plain text."""
    if not result_holder.stdout:
        result_holder.stdout = []
    if output_message:
        result_holder.stdout.append(output_message)
    if resource_output:
        result_holder.stdout.append(resource_output)
    if raw_output:
        result_holder.stdout.append(raw_output)