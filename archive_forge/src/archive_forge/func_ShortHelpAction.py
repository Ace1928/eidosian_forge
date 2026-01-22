from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def ShortHelpAction(command):
    """Get an argparse.Action that prints a short help.

  Args:
    command: calliope._CommandCommon, The command object that we're helping.

  Returns:
    argparse.Action, the action to use.
  """

    def Func():
        metrics.Help(command.dotted_name, '-h')
        log.out.write(command.GetUsage())
    return FunctionExitAction(Func)