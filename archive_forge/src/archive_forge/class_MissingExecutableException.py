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
class MissingExecutableException(BinaryOperationError):
    """Raised if an executable can not be found on the path."""

    def __init__(self, exec_name, custom_message=None):
        if custom_message:
            error_msg = custom_message
        else:
            error_msg = _DEFAULT_MISSING_EXEC_MESSAGE.format(exec_name)
        super(MissingExecutableException, self).__init__(error_msg)