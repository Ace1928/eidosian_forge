from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
def Visible(cmd_class):
    """Decorator for making visible calliope commands and groups.

  Decorate a subclass of base.Command or base.Group with this function, and the
  decorated command or group will show up in help text. This is the default
  for base.Command and base.Group subclasses.

  Args:
    cmd_class: base._Common, A calliope command or group.

  Returns:
    A modified version of the provided class.
  """
    cmd_class._is_hidden = False
    return cmd_class