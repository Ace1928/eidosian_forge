from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def MayHaveUpdates(self):
    """Returns true if this Diff is non-empty."""
    return any([self._additions, self._subtractions, self._clear])