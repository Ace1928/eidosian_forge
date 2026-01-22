from __future__ import unicode_literals
import os
import re
import six
import termios
import tty
from six.moves import range
from ..keys import Keys
from ..key_binding.input_processor import KeyPress
class _IsPrefixOfLongerMatchCache(dict):
    """
    Dictiory that maps input sequences to a boolean indicating whether there is
    any key that start with this characters.
    """

    def __missing__(self, prefix):
        if _cpr_response_prefix_re.match(prefix) or _mouse_event_prefix_re.match(prefix):
            result = True
        else:
            result = any((v for k, v in ANSI_SEQUENCES.items() if k.startswith(prefix) and k != prefix))
        self[prefix] = result
        return result