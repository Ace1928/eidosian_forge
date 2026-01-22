from __future__ import (absolute_import, division, print_function)
import os
import sys
import yaml
import ovirtsdk4 as sdk
from ovirtsdk4 import types
from bcolors import bcolors
from configparser import ConfigParser
def _set_dr_conf_variables(self, conf_file):
    _SECTION = 'validate_vars'
    _VAR_FILE = 'var_file'
    settings = ConfigParser()
    settings.read(conf_file)
    if _SECTION not in settings.sections():
        settings.add_section(_SECTION)
    if not settings.has_option(_SECTION, _VAR_FILE):
        settings.set(_SECTION, _VAR_FILE, '')
    var_file = settings.get(_SECTION, _VAR_FILE, vars=DefaultOption(settings, _SECTION, site=self.def_var_file))
    var_file = os.path.expanduser(var_file)
    while not os.path.isfile(var_file):
        var_file = input("%s%sVar file '%s' does not exist. Please provide the location of the var file (%s): %s" % (WARN, PREFIX, var_file, self.def_var_file, END)) or self.def_var_file
        var_file = os.path.expanduser(var_file)
    self.var_file = var_file
    self.primary_pwd = input('%s%sPlease provide password for the primary setup: %s' % (INPUT, PREFIX, END))
    self.second_pwd = input('%s%sPlease provide password for the secondary setup: %s' % (INPUT, PREFIX, END))