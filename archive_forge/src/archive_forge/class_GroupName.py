from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class GroupName(Expression):
    """A glyph group"""

    def __init__(self, group, parser, location=None):
        Expression.__init__(self, location)
        self.group = group
        self.parser_ = parser

    def glyphSet(self, groups=None):
        group = self.parser_.resolve_group(self.group)
        if group is not None:
            self.glyphs_ = group.glyphSet(groups)
            return self.glyphs_
        else:
            raise VoltLibError('Group "%s" is used but undefined.' % self.group, self.location)

    def __str__(self):
        return f' GROUP "{self.group}"'