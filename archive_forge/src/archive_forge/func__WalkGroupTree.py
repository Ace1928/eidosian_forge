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
def _WalkGroupTree(group):
    """Visits each group in the CLI group tree.

  Args:
    group: backend.CommandGroup, root CLI subgroup node.
  Yields:
    group instance.
  """
    yield group
    for sub_group in six.itervalues(group.groups):
        for value in _WalkGroupTree(sub_group):
            yield value