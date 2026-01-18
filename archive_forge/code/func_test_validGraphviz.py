from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_validGraphviz(self):
    """
        L{graphviz} emits valid graphviz data.
        """
    p = subprocess.Popen('dot', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = p.communicate(''.join(sampleMachine().asDigraph()).encode('utf-8'))
    self.assertEqual(p.returncode, 0)