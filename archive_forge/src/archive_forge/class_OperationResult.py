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
class OperationResult(object):
    """Generic Holder for Operation return values and errors."""

    def __init__(self, command_str, output=None, errors=None, status=0, failed=False, execution_context=None):
        self.executed_command = command_str
        self.stdout = output
        self.stderr = errors
        self.exit_code = status
        self.context = execution_context
        self.failed = failed

    def __str__(self):
        output = collections.OrderedDict()
        output['executed_command'] = self.executed_command
        output['stdout'] = self.stdout
        output['stderr'] = self.stderr
        output['exit_code'] = self.exit_code
        output['failed'] = self.failed
        output['execution_context'] = self.context
        return yaml.dump(output)

    def __eq__(self, other):
        if isinstance(other, BinaryBackedOperation.OperationResult):
            return self.executed_command == other.executed_command and self.stdout == other.stdout and (self.stderr == other.stderr) and (self.exit_code == other.exit_code) and (self.failed == other.failed) and (self.context == other.context)
        return False

    def __repr__(self):
        return self.__str__()