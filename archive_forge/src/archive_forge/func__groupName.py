import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _groupName(self, group):
    try:
        name = group.group
    except AttributeError:
        name = group
    return ast.GlyphClassName(self._glyphclasses[name.lower()])