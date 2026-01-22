from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
class LintError(object):
    """Validation failure.

  Attributes:
    name: str, The name of the validation that produced this failure.
    command: calliope.backend.CommandCommon, The offending command.
    msg: str, A message indicating what the problem was.
  """

    def __init__(self, name, command, error_message):
        self.name = name
        self.command = command
        self.msg = '[{cmd}]: {msg}'.format(cmd='.'.join(command.GetPath()), msg=error_message)