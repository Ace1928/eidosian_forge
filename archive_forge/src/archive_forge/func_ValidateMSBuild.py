import re
import sys
def ValidateMSBuild(self, value):
    if value not in self._msbuild_values:
        raise ValueError('unrecognized enumerated value %s' % value)