from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
class ScriptStatement(Statement):
    """A ``script`` statement."""

    def __init__(self, script, location=None):
        Statement.__init__(self, location)
        self.script = script

    def build(self, builder):
        """Calls the builder's ``set_script`` callback."""
        builder.set_script(self.location, self.script)

    def asFea(self, indent=''):
        return 'script {};'.format(self.script.strip())