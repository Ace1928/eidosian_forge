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
def test_setValue_block_implicit(self):
    _test = {'scenario': {'merlion': True, 'detection': [1]}, 'foo': 1}
    ref = self._reference
    ref['scenario'].update(_test['scenario'])
    ref['foo'] = 1
    self.config.set_value(_test)
    self.assertEqual(ref, self.config.value())
    _test = {'scenario': {'merlion': True, 'detection': [1]}, 'bar': 1}
    ref['bar'] = 1
    self.config.set_value(_test)
    self.assertEqual(ref, self.config.value())