import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def writeSourceInto(self, source, path, moduleName):
    directory = self.FilePath(path)
    module = directory.child(moduleName)
    with open(module.path, 'w') as f:
        f.write(textwrap.dedent(source))
    return self.PythonPath([directory.path])