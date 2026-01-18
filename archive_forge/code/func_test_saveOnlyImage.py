from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_saveOnlyImage(self):
    """
        Passing an empty string for --dot-directory/-d disables saving dot
        files.
        """
    for arg in ('--dot-directory', '-d'):
        self.digraphRecorder.reset()
        self.collectedOutput = []
        self.tool(argv=[self.fakeFQPN, arg, ''])
        self.assertFalse(any(('dot' in line for line in self.collectedOutput)))
        self.assertEqual(len(self.digraphRecorder.renderCalls), 1)
        call, = self.digraphRecorder.renderCalls
        self.assertEqual('{}.dot'.format(self.fakeFQPN), call['filename'])
        self.assertTrue(call['cleanup'])
        self.assertFalse(self.digraphRecorder.saveCalls)