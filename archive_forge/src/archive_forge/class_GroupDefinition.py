from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class GroupDefinition(Statement):

    def __init__(self, name, enum, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.enum = enum
        self.glyphs_ = None

    def glyphSet(self, groups=None):
        if groups is not None and self.name in groups:
            raise VoltLibError('Group "%s" contains itself.' % self.name, self.location)
        if self.glyphs_ is None:
            if groups is None:
                groups = set({self.name})
            else:
                groups.add(self.name)
            self.glyphs_ = self.enum.glyphSet(groups)
        return self.glyphs_

    def __str__(self):
        enum = self.enum and str(self.enum) or ''
        return f'DEF_GROUP "{self.name}"\n{enum}\nEND_GROUP'