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
def FunctionExitAction(func):
    """Get an argparse.Action that runs the provided function, and exits.

  Args:
    func: func, the function to execute.

  Returns:
    argparse.Action, the action to use.
  """

    class Action(argparse.Action):
        """The action created for FunctionExitAction."""

        def __init__(self, **kwargs):
            kwargs['nargs'] = 0
            super(Action, self).__init__(**kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            base.LogCommand(parser.prog, namespace)
            metrics.Loaded()
            func()
            sys.exit(0)
    return Action