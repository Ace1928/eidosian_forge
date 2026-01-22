import errno
import operator
import os
import random
import re
import shlex
import socket
import stat
import string
import struct
import sys
import textwrap
import time
import traceback
from functools import reduce
from os import path
from typing import Optional
from twisted.internet import protocol, reactor, task
from twisted.persisted import styles
from twisted.protocols import basic
from twisted.python import _textattributes, log, reflect
class _CharacterAttributes(_textattributes.CharacterAttributesMixin):
    """
    Factory for character attributes, including foreground and background color
    and non-color attributes such as bold, reverse video and underline.

    Character attributes are applied to actual text by using object
    indexing-syntax (C{obj['abc']}) after accessing a factory attribute, for
    example::

        attributes.bold['Some text']

    These can be nested to mix attributes::

        attributes.bold[attributes.underline['Some text']]

    And multiple values can be passed::

        attributes.normal[attributes.bold['Some'], ' text']

    Non-color attributes can be accessed by attribute name, available
    attributes are:

        - bold
        - reverseVideo
        - underline

    Available colors are:

        0. white
        1. black
        2. blue
        3. green
        4. light red
        5. red
        6. magenta
        7. orange
        8. yellow
        9. light green
        10. cyan
        11. light cyan
        12. light blue
        13. light magenta
        14. gray
        15. light gray

    @ivar fg: Foreground colors accessed by attribute name, see above
        for possible names.

    @ivar bg: Background colors accessed by attribute name, see above
        for possible names.

    @since: 13.1
    """
    fg = _textattributes._ColorAttribute(_textattributes._ForegroundColorAttr, _IRC_COLORS)
    bg = _textattributes._ColorAttribute(_textattributes._BackgroundColorAttr, _IRC_COLORS)
    attrs = {'bold': _BOLD, 'reverseVideo': _REVERSE_VIDEO, 'underline': _UNDERLINE}