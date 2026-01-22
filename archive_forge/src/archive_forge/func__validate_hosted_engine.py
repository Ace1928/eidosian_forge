from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _validate_hosted_engine(self, var_file):
    domains = var_file[self.domain_map]
    hosted = 'hosted_storage'
    for domain in domains:
        primary = domain['dr_primary_name']
        secondary = domain['dr_secondary_name']
        if primary == hosted or secondary == hosted:
            print('%s%sHosted storage domains are not supported.%s' % (FAIL, PREFIX, END))
            return False
    return True