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
@document_kwargs_from_configdict('CONFIG')
class ExampleClass(object):
    CONFIG = ExampleConfig()

    @document_kwargs_from_configdict(CONFIG)
    def __init__(self):
        """A simple docstring"""

    @document_kwargs_from_configdict(CONFIG, doc='A simple docstring\n', visibility=USER_OPTION)
    def fcn(self):
        pass