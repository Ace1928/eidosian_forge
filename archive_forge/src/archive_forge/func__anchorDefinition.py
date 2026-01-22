import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _anchorDefinition(self, anchordef):
    anchorname = anchordef.name
    glyphname = anchordef.glyph_name
    anchor = self._anchor(anchordef.pos)
    if anchorname.startswith('MARK_'):
        name = '_'.join(anchorname.split('_')[1:])
        markclass = ast.MarkClass(self._className(name))
        glyph = self._glyphName(glyphname)
        markdef = MarkClassDefinition(markclass, anchor, glyph)
        self._markclasses[glyphname, anchorname] = markdef
    else:
        if glyphname not in self._anchors:
            self._anchors[glyphname] = {}
        if anchorname not in self._anchors[glyphname]:
            self._anchors[glyphname][anchorname] = {}
        self._anchors[glyphname][anchorname][anchordef.component] = anchor