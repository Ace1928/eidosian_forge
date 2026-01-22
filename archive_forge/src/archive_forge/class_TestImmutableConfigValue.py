import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
class TestImmutableConfigValue(unittest.TestCase):

    def test_immutable_config_value(self):
        config = ConfigDict()
        config.declare('a', ConfigValue(default=1, domain=int))
        config.declare('b', ConfigValue(default=1, domain=int))
        config.a = 2
        config.b = 3
        self.assertEqual(config.a, 2)
        self.assertEqual(config.b, 3)
        locker = MarkImmutable(config.get('a'), config.get('b'))
        with self.assertRaisesRegex(RuntimeError, 'is currently immutable'):
            config.a = 4
        with self.assertRaisesRegex(RuntimeError, 'is currently immutable'):
            config.b = 5
        config.a = 2
        config.b = 3
        self.assertEqual(config.a, 2)
        self.assertEqual(config.b, 3)
        locker.release_lock()
        config.a = 4
        config.b = 5
        self.assertEqual(config.a, 4)
        self.assertEqual(config.b, 5)
        with self.assertRaisesRegex(ValueError, 'Only ConfigValue instances can be marked immutable'):
            locker = MarkImmutable(config.get('a'), config.b)
        self.assertEqual(type(config.get('a')), ConfigValue)
        config.a = 6
        self.assertEqual(config.a, 6)
        config.declare('c', ConfigValue(default=-1, domain=int))
        locker = MarkImmutable(config.get('a'), config.get('b'))
        config2 = config({'c': -2})
        self.assertEqual(config2.a, 6)
        self.assertEqual(config2.b, 5)
        self.assertEqual(config2.c, -2)
        self.assertIs(type(config2.get('a')), ConfigValue)
        self.assertIs(type(config2.get('b')), ConfigValue)
        self.assertIs(type(config2.get('c')), ConfigValue)
        config.set_value(config2)
        self.assertEqual(config.a, 6)
        self.assertEqual(config.b, 5)
        self.assertEqual(config.c, -2)
        self.assertIs(type(config.get('a')), ImmutableConfigValue)
        self.assertIs(type(config.get('b')), ImmutableConfigValue)
        self.assertIs(type(config.get('c')), ConfigValue)
        config3 = config({'a': 1})
        self.assertEqual(config3.a, 1)
        self.assertEqual(config3.b, 5)
        self.assertEqual(config3.c, -2)
        self.assertIs(type(config3.get('a')), ConfigValue)
        self.assertIs(type(config3.get('b')), ConfigValue)
        self.assertIs(type(config3.get('c')), ConfigValue)
        with self.assertRaisesRegex(RuntimeError, ' is currently immutable'):
            config.set_value(config3)
        locker.release_lock()
        config.reset()
        self.assertEqual(config.a, 1)
        self.assertEqual(config.b, 1)
        with locker:
            config.reset()
            self.assertEqual(config.a, 1)
            self.assertEqual(config.b, 1)
        config.a = 2
        with locker:
            with self.assertRaisesRegex(RuntimeError, 'is currently immutable'):
                config.reset()