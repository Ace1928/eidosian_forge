from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _create_stylesheet(self):
    t2c = self.ttype2class = {Token: ''}
    c2s = self.class2style = {}
    for ttype, ndef in self.style:
        name = self._get_css_class(ttype)
        style = ''
        if ndef['color']:
            style += 'color: #%s; ' % ndef['color']
        if ndef['bold']:
            style += 'font-weight: bold; '
        if ndef['italic']:
            style += 'font-style: italic; '
        if ndef['underline']:
            style += 'text-decoration: underline; '
        if ndef['bgcolor']:
            style += 'background-color: #%s; ' % ndef['bgcolor']
        if ndef['border']:
            style += 'border: 1px solid #%s; ' % ndef['border']
        if style:
            t2c[ttype] = name
            c2s[name] = (style[:-2], ttype, len(ttype))