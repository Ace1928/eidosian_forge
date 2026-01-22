import os
import re
import shlex
import sys
import pygments
from pathlib import Path
from IPython.utils.text import marquee
from IPython.utils import openpy
from IPython.utils import py3compat
class ClearMixin(object):
    """Use this mixin to make Demo classes with less visual clutter.

    Demos using this mixin will clear the screen before every block and use
    blank marquees.

    Note that in order for the methods defined here to actually override those
    of the classes it's mixed with, it must go /first/ in the inheritance
    tree.  For example:

        class ClearIPDemo(ClearMixin,IPythonDemo): pass

    will provide an IPythonDemo class with the mixin's features.
    """

    def marquee(self, txt='', width=78, mark='*'):
        """Blank marquee that returns '' no matter what the input."""
        return ''

    def pre_cmd(self):
        """Method called before executing each block.

        This one simply clears the screen."""
        from IPython.utils.terminal import _term_clear
        _term_clear()