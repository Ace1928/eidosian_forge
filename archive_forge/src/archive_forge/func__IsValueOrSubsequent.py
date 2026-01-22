from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def _IsValueOrSubsequent(c):
    """Returns True if c is a valid value or subsequent (not first) character."""
    return c in ('_', '-') or c.isdigit() or _IsLower(c)