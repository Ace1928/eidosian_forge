from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Any
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
import six
def _Walk(node, parent):
    """Walk() helper that calls self.Visit() on each node in the CLI tree.

      Args:
        node: CommandCommon tree node.
        parent: The parent Visit() return value, None at the top level.

      Returns:
        The return value of the outer Visit() call.
      """
    if not node.is_group:
        self._Visit(node, parent, is_group=False)
        return parent
    parent = self._Visit(node, parent, is_group=True)
    commands_and_groups = []
    if node.commands:
        for name, command in six.iteritems(node.commands):
            if _Include(command):
                commands_and_groups.append((name, command, False))
    if node.groups:
        for name, command in six.iteritems(node.groups):
            if _Include(command, traverse=True):
                commands_and_groups.append((name, command, True))
    for _, command, is_group in sorted(commands_and_groups):
        if is_group:
            _Walk(command, parent)
        else:
            self._Visit(command, parent, is_group=False)
    return parent