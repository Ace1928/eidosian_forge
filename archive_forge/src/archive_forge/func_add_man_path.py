import os
import shlex
import sys
from pbr import find_package
from pbr.hooks import base
def add_man_path(self, man_path):
    self.data_files = "%s\n'%s' =" % (self.data_files, man_path)