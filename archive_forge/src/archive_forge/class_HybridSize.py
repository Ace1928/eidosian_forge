import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class HybridSize(object):
    """The size associated with a hybrid function."""
    configsizes = FormulaConfig.hybridsizes

    def getsize(self, function):
        """Read the size for a function and parse it."""
        sizestring = self.configsizes[function.command]
        for name in function.params:
            if name in sizestring:
                size = function.params[name].value.computesize()
                sizestring = sizestring.replace(name, str(size))
        if '$' in sizestring:
            Trace.error('Unconverted variable in hybrid size: ' + sizestring)
            return 1
        return eval(sizestring)