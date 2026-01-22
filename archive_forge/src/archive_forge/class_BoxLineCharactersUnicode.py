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
class BoxLineCharactersUnicode(BoxLineCharacters):
    """unicode Box/line drawing characters (cp437 compatible unicode)."""
    dl = '┐'
    dr = '┌'
    h = '─'
    hd = '┬'
    hu = '┴'
    ul = '┘'
    ur = '└'
    v = '│'
    vh = '┼'
    vl = '┤'
    vr = '├'
    d_dl = '╗'
    d_dr = '╔'
    d_h = '═'
    d_hd = '╦'
    d_hu = '╩'
    d_ul = '╝'
    d_ur = '╚'
    d_v = '║'
    d_vh = '╬'
    d_vl = '╣'
    d_vr = '╠'