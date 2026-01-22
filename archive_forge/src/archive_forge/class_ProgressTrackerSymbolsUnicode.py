from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import locale
import os
import sys
import unicodedata
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.console.style import text
from googlecloudsdk.core.util import encoding as encoding_util
import six
class ProgressTrackerSymbolsUnicode(ProgressTrackerSymbols):
    """Characters used by progress trackers."""

    @property
    def spin_marks(self):
        return ['⠏', '⠛', '⠹', '⠼', '⠶', '⠧']
    success = text.TypedText(['✓'], text_type=text.TextTypes.PT_SUCCESS)
    failed = text.TypedText(['X'], text_type=text.TextTypes.PT_FAILURE)
    interrupted = '-'
    not_started = '.'
    prefix_length = 2