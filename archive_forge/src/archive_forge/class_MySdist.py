from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import os
import datetime
import traceback
import platform  # NOQA
from _ast import *  # NOQA
from ast import parse  # NOQA
from setuptools import setup, Extension, Distribution  # NOQA
from setuptools.command import install_lib  # NOQA
from setuptools.command.sdist import sdist as _sdist  # NOQA
class MySdist(_sdist):

    def initialize_options(self):
        _sdist.initialize_options(self)
        dist_base = os.environ.get('PYDISTBASE')
        fpn = getattr(getattr(self, 'nsp', self), 'full_package_name', None)
        if fpn and dist_base:
            print('setting  distdir {}/{}'.format(dist_base, fpn))
            self.dist_dir = os.path.join(dist_base, fpn)