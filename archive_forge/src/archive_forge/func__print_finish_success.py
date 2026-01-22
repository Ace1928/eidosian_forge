from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _print_finish_success(self):
    print('%s%sFinished validation of variable mapping file for oVirt ansible disaster recovery%s' % (INFO, PREFIX, END))