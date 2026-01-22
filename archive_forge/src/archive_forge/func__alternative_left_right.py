import re
import time
import platform
from collections import OrderedDict
import six
def _alternative_left_right(term):
    """
    Determine and return mapping of left and right arrow keys sequences.

    :arg blessed.Terminal term: :class:`~.Terminal` instance.
    :rtype: dict
    :returns: Dictionary of sequences ``term._cuf1``, and ``term._cub1``,
        valued as ``KEY_RIGHT``, ``KEY_LEFT`` (when appropriate).

    This function supports :func:`get_terminal_sequences` to discover
    the preferred input sequence for the left and right application keys.

    It is necessary to check the value of these sequences to ensure we do not
    use ``u' '`` and ``u'\\b'`` for ``KEY_RIGHT`` and ``KEY_LEFT``,
    preferring their true application key sequence, instead.
    """
    keymap = {}
    if term._cuf1 and term._cuf1 != u' ':
        keymap[term._cuf1] = curses.KEY_RIGHT
    if term._cub1 and term._cub1 != u'\x08':
        keymap[term._cub1] = curses.KEY_LEFT
    return keymap