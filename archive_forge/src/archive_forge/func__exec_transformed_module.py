import sys
import logging
import os
import copy
from lib2to3.pgen2.parse import ParseError
from lib2to3.refactor import RefactoringTool
from libfuturize import fixes
def _exec_transformed_module(self, module):
    source = self.get_source(self.name)
    pathname = self.path
    if detect_python2(source, pathname):
        source = transform(source, pathname)
    code = compile(source, pathname, 'exec')
    exec(code, module.__dict__)