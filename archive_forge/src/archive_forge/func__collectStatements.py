import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _collectStatements(self, doc, tables):
    groups = [s for s in doc.statements if isinstance(s, VAst.GroupDefinition)]
    for statement in sorted(groups, key=lambda x: Group(x)):
        self._groupDefinition(statement)
    for statement in doc.statements:
        if isinstance(statement, VAst.GlyphDefinition):
            self._glyphDefinition(statement)
        elif isinstance(statement, VAst.AnchorDefinition):
            if 'GPOS' in tables:
                self._anchorDefinition(statement)
        elif isinstance(statement, VAst.SettingDefinition):
            self._settingDefinition(statement)
        elif isinstance(statement, VAst.GroupDefinition):
            pass
        elif isinstance(statement, VAst.ScriptDefinition):
            self._scriptDefinition(statement)
        elif not isinstance(statement, VAst.LookupDefinition):
            raise NotImplementedError(statement)
    for statement in doc.statements:
        if isinstance(statement, VAst.LookupDefinition):
            if statement.pos and 'GPOS' not in tables:
                continue
            if statement.sub and 'GSUB' not in tables:
                continue
            self._lookupDefinition(statement)