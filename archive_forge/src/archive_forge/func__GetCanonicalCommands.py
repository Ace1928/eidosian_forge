from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
import re
from googlecloudsdk.command_lib.static_completion import lookup
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _GetCanonicalCommands(tree):
    """Return list of all canonical commands in CLI tree in arbitrary order.

  Args:
    tree: The root of the tree that will be traversed to find commands.

  Returns:
    [[canonical_command_words]]: List of lists, all possible sequences of
      canonical command words in the tree.
  """
    results = []
    _GetCanonicalCommandsHelper(tree, results, prefix=[])
    return results