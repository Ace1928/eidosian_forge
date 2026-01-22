from __future__ import absolute_import
import sys
import subprocess
import time
from twisted.trial.unittest import TestCase
from crochet._shutdown import (
from ..tests import crochet_directory
import threading, sys
from crochet._shutdown import register, _watchdog
class FunctionRegistryTests(TestCase):
    """
    Tests for FunctionRegistry.
    """

    def test_called(self):
        """
        Functions registered with a FunctionRegistry are called in reverse
        order by run().
        """
        result = []
        registry = FunctionRegistry()
        registry.register(lambda: result.append(1))
        registry.register(lambda x: result.append(x), 2)
        registry.register(lambda y: result.append(y), y=3)
        registry.run()
        self.assertEqual(result, [3, 2, 1])

    def test_log_errors(self):
        """
        Registered functions that raise an error have the error logged, and
        run() continues processing.
        """
        result = []
        registry = FunctionRegistry()
        registry.register(lambda: result.append(2))
        registry.register(lambda: 1 / 0)
        registry.register(lambda: result.append(1))
        registry.run()
        self.assertEqual(result, [1, 2])
        excs = self.flushLoggedErrors(ZeroDivisionError)
        self.assertEqual(len(excs), 1)