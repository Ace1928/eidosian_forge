import logging
import re
from io import StringIO
from fontTools.feaLib import ast
from fontTools.ttLib import TTFont, TTLibError
from fontTools.voltLib import ast as VAst
from fontTools.voltLib.parser import Parser as VoltParser
def _settingDefinition(self, setting):
    if setting.name.startswith('COMPILER_'):
        self._settings[setting.name] = setting.value
    else:
        log.warning(f'Unsupported setting ignored: {setting.name}')