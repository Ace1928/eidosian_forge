from __future__ import unicode_literals
import os
import re
import six
import termios
import tty
from six.moves import range
from ..keys import Keys
from ..key_binding.input_processor import KeyPress
@classmethod
def _patch_iflag(cls, attrs):
    return attrs | termios.ICRNL