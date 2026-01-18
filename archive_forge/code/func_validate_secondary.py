from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def validate_secondary(self):
    isValid = True
    if self.second_url is None:
        print(self.error_msg % (FAIL, PREFIX, 'url', 'secondary', END))
        isValid = False
    if self.second_user is None:
        print(self.error_msg % (FAIL, PREFIX, 'username', 'secondary', END))
        isValid = False
    if self.second_ca is None:
        print(self.error_msg % (FAIL, PREFIX, 'ca', 'secondary', END))
        isValid = False
    return isValid