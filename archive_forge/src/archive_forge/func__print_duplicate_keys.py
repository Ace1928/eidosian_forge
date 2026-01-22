from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _print_duplicate_keys(self, duplicates, keys):
    ret_val = False
    for key in keys:
        if len(duplicates[key]) > 0:
            print('%s%sFound the following duplicate keys in %s: %s%s' % (FAIL, PREFIX, key, list(duplicates[key]), END))
            ret_val = True
    return ret_val