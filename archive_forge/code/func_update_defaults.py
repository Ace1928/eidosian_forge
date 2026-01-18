from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def update_defaults(self, new_defaults, overwrite=True):
    for key, value in new_defaults.items():
        if not overwrite and key in self.parser._defaults:
            continue
        self.parser._defaults[key] = value