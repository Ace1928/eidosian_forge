from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _read_var_file(self):
    with open(self.var_file, 'r') as info:
        info_dict = yaml.safe_load(info)
    return info_dict