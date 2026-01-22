import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
class PBXShellScriptBuildPhase(XCBuildPhase):
    _schema = XCBuildPhase._schema.copy()
    _schema.update({'inputPaths': [1, str, 0, 1, []], 'name': [0, str, 0, 0], 'outputPaths': [1, str, 0, 1, []], 'shellPath': [0, str, 0, 1, '/bin/sh'], 'shellScript': [0, str, 0, 1], 'showEnvVarsInLog': [0, int, 0, 0]})

    def Name(self):
        if 'name' in self._properties:
            return self._properties['name']
        return 'ShellScript'