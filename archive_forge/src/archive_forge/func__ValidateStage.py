from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
import signal
import sys
import threading
import time
import enum
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import multiline
from googlecloudsdk.core.console.style import parser
import six
def _ValidateStage(self, key, allow_complete=False):
    """Validates the stage belongs to the tracker.

    Args:
      key: the key of the stage to validate.
      allow_complete: whether to error on an already-complete stage

    Returns:
      The validated Stage object, even if we were passed a key.
    """
    if key not in self:
        raise ValueError('This stage does not belong to this progress tracker.')
    stage = self.get(key)
    if not allow_complete and stage.status not in {StageCompletionStatus.NOT_STARTED, StageCompletionStatus.RUNNING}:
        raise ValueError('This stage has already completed.')
    return stage