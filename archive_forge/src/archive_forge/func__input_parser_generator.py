from __future__ import unicode_literals
import os
import re
import six
import termios
import tty
from six.moves import range
from ..keys import Keys
from ..key_binding.input_processor import KeyPress
def _input_parser_generator(self):
    """
        Coroutine (state machine) for the input parser.
        """
    prefix = ''
    retry = False
    flush = False
    while True:
        flush = False
        if retry:
            retry = False
        else:
            c = (yield)
            if c == _Flush:
                flush = True
            else:
                prefix += c
        if prefix:
            is_prefix_of_longer_match = _IS_PREFIX_OF_LONGER_MATCH_CACHE[prefix]
            match = self._get_match(prefix)
            if (flush or not is_prefix_of_longer_match) and match:
                self._call_handler(match, prefix)
                prefix = ''
            elif (flush or not is_prefix_of_longer_match) and (not match):
                found = False
                retry = True
                for i in range(len(prefix), 0, -1):
                    match = self._get_match(prefix[:i])
                    if match:
                        self._call_handler(match, prefix[:i])
                        prefix = prefix[i:]
                        found = True
                if not found:
                    self._call_handler(prefix[0], prefix[0])
                    prefix = prefix[1:]