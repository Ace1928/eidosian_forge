from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
import six
def ShouldPrompt(self):
    """Checks whether to prompt or not."""
    if not (log.out.isatty() and log.err.isatty()):
        return False
    last_prompt_time = self.record.last_prompt_time
    now = time.time()
    if last_prompt_time and now - last_prompt_time < self.PROMPT_INTERVAL:
        return False
    return True