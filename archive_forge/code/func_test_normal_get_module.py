import os
import sys
from breezy import branch, osutils, registry, tests
def test_normal_get_module(self):

    class AThing:
        """Something"""
    a_registry = registry.Registry()
    a_registry.register('obj', AThing())
    self.assertEqual('breezy.tests.test_registry', a_registry._get_module('obj'))