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
def test_setValue_scalar_badDomain(self):
    with self.assertRaisesRegex(ValueError, 'invalid value for configuration'):
        self.config['flushing']['flush nodes']['rate'] = 'a'
    val = self.config['flushing']['flush nodes']['rate']
    self.assertIs(type(val), float)
    self.assertEqual(val, 600.0)