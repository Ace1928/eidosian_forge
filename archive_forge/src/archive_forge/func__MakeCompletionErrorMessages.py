from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.cache import resource_cache
from googlecloudsdk.core.console import console_attr
import six
@classmethod
def _MakeCompletionErrorMessages(cls, msgs):
    """Returns a msgs list that will display 1 per line as completions."""
    attr = console_attr.GetConsoleAttr()
    width, _ = attr.GetTermSize()
    return [msg + (width // 2 - len(msg)) * ' ' for msg in msgs]