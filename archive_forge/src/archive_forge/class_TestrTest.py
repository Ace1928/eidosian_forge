from __future__ import unicode_literals
from distutils.command import install as du_install
from distutils import log
import email
import email.errors
import os
import re
import sys
import warnings
import pkg_resources
import setuptools
from setuptools.command import develop
from setuptools.command import easy_install
from setuptools.command import egg_info
from setuptools.command import install
from setuptools.command import install_scripts
from setuptools.command import sdist
from pbr import extra_files
from pbr import git
from pbr import options
import pbr.pbr_json
from pbr import testr_command
from pbr import version
import threading
from %(module_name)s import %(import_target)s
import sys
from %(module_name)s import %(import_target)s
class TestrTest(testr_command.Testr):
    """Make setup.py test do the right thing."""
    command_name = 'test'
    description = 'DEPRECATED: Run unit tests using testr'

    def run(self):
        warnings.warn('testr integration is deprecated in pbr 4.2 and will be removed in a future release. Please call your test runner directly', DeprecationWarning)
        testr_command.Testr.run(self)