from __future__ import (absolute_import, division, print_function)
import logging
import os.path
import subprocess
from subprocess import call
import sys
import time
from configparser import ConfigParser
from bcolors import bcolors
class DefaultOption(dict):

    def __init__(self, config, section, **kv):
        self._config = config
        self._section = section
        dict.__init__(self, **kv)

    def items(self):
        _items = []
        for option in self:
            if not self._config.has_option(self._section, option):
                _items.append((option, self[option]))
            else:
                value_in_config = self._config.get(self._section, option)
                _items.append((option, value_in_config))
        return _items