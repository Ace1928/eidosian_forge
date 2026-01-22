import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _scriptDefinition(self, script):
    stag = script.tag
    for lang in script.langs:
        ltag = lang.tag
        for feature in lang.features:
            lookups = {l.split('\\')[0]: True for l in feature.lookups}
            ftag = feature.tag
            if ftag not in self._features:
                self._features[ftag] = {}
            if stag not in self._features[ftag]:
                self._features[ftag][stag] = {}
            assert ltag not in self._features[ftag][stag]
            self._features[ftag][stag][ltag] = lookups.keys()