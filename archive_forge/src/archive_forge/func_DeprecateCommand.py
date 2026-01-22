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
def DeprecateCommand(cmd_class):
    """Wrapper Function that creates actual decorated class.

    Args:
      cmd_class: base.Command or base.Group subclass to be decorated

    Returns:
      The decorated class.
    """
    if is_removed:
        msg = error
        deprecation_tag = '{0}(REMOVED){0} '.format(MARKDOWN_BOLD)
    else:
        msg = warning
        deprecation_tag = '{0}(DEPRECATED){0} '.format(MARKDOWN_BOLD)
    cmd_class.AddNotice(deprecation_tag, msg)
    cmd_class.SetDeprecated(True)

    def RunDecorator(run_func):

        @wraps(run_func)
        def WrappedRun(*args, **kw):
            if is_removed:
                raise DeprecationException(error)
            log.warning(warning)
            return run_func(*args, **kw)
        return WrappedRun
    if issubclass(cmd_class, Group):
        cmd_class.Filter = RunDecorator(cmd_class.Filter)
    else:
        cmd_class.Run = RunDecorator(cmd_class.Run)
    if is_removed:
        return Hidden(cmd_class)
    return cmd_class