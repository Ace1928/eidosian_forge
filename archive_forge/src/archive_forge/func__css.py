from __future__ import absolute_import
import os
import os.path
import re
import codecs
import textwrap
from datetime import datetime
from functools import partial
from collections import defaultdict
from xml.sax.saxutils import escape as html_escape
from . import Version
from .Code import CCodeWriter
from .. import Utils
def _css(self):
    """css template will later allow to choose a colormap"""
    css = [self._css_template]
    for i in range(255):
        color = u'FFFF%02x' % int(255.0 // (1.0 + i / 10.0))
        css.append('.cython.score-%d {background-color: #%s;}' % (i, color))
    try:
        from pygments.formatters import HtmlFormatter
    except ImportError:
        pass
    else:
        css.append(HtmlFormatter().get_style_defs('.cython'))
    return '\n'.join(css)