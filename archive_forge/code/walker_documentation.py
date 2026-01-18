from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Any
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
import six
Walk() helper that calls self.Visit() on each node in the CLI tree.

      Args:
        node: CommandCommon tree node.
        parent: The parent Visit() return value, None at the top level.

      Returns:
        The return value of the outer Visit() call.
      