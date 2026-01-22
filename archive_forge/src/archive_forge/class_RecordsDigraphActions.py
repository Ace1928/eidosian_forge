from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
class RecordsDigraphActions(object):
    """
    Records calls made to L{FakeDigraph}.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.renderCalls = []
        self.saveCalls = []