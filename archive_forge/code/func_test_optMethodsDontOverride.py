import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_optMethodsDontOverride(self):
    """
        opt_* methods on Options classes should not override the
        data provided in optFlags or optParameters.
        """

    class Options(usage.Options):
        optFlags = [['flag', 'f', 'A flag']]
        optParameters = [['param', 'p', None, 'A param']]

        def opt_flag(self):
            """junk description"""

        def opt_param(self, param):
            """junk description"""
    opts = Options()
    argGen = _shellcomp.ZshArgumentsGenerator(opts, 'ace', None)
    self.assertEqual(argGen.getDescription('flag'), 'A flag')
    self.assertEqual(argGen.getDescription('param'), 'A param')