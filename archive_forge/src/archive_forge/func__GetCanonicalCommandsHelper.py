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
def _GetCanonicalCommandsHelper(tree, results, prefix):
    """Helper method to _GetCanonicalCommands.

  Args:
    tree: The root of the tree that will be traversed to find commands.
    results: The results list to append to.
    prefix: [str], the canonical command line words so far. Once we reach
      a leaf node, prefix contains a canonical command and a copy is
      appended to results.

  Returns:
    None
  """
    if not tree.get(lookup.LOOKUP_COMMANDS):
        results.append(prefix[:])
        return
    for command, command_tree in six.iteritems(tree[lookup.LOOKUP_COMMANDS]):
        prefix.append(command)
        _GetCanonicalCommandsHelper(command_tree, results, prefix)
        prefix.pop()