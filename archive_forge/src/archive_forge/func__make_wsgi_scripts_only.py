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
def _make_wsgi_scripts_only(self, dist, executable):
    try:
        header = easy_install.ScriptWriter.get_header('', executable)
    except AttributeError:
        header = easy_install.get_script_header('', executable)
    wsgi_script_template = ENTRY_POINTS_MAP['wsgi_scripts']
    for name, ep in dist.get_entry_map('wsgi_scripts').items():
        content = generate_script('wsgi_scripts', ep, header, wsgi_script_template)
        self.write_script(name, content)