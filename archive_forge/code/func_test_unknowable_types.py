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
def test_unknowable_types(self):
    obj = ConfigValue()

    def local_fcn():
        pass
    try:
        pickle.dumps(local_fcn)
        local_picklable = True
    except:
        local_picklable = False
    self.assertIs(_picklable(_display, obj), _display)
    if local_picklable:
        self.assertIs(_picklable(local_fcn, obj), local_fcn)
    else:
        self.assertIsNot(_picklable(local_fcn, obj), local_fcn)
    self.assertIs(_picklable(_display, obj), _display)
    if local_picklable:
        self.assertIs(_picklable(local_fcn, obj), local_fcn)
    else:
        self.assertIsNot(_picklable(local_fcn, obj), local_fcn)
    self.assertIn(types.FunctionType, _picklable.unknowable_types)
    self.assertNotIn(types.FunctionType, _picklable.known)