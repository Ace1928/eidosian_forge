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
def test_value_ConfigList_complexPopulated(self):
    self.config['scenarios'].append()
    val = self.config['scenarios'].value()
    self.assertIs(type(val), list)
    self.assertEqual(len(val), 1)
    self.assertEqual(val, [{'detection': [1, 2, 3], 'merlion': False, 'scenario file': 'Net3.tsg'}])