from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Any
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
import six
def _Visit(self, node, parent, is_group):
    self._num_visited += 1
    self._progress_callback(self._num_visited // self._num_nodes)
    return self.Visit(node, parent, is_group)