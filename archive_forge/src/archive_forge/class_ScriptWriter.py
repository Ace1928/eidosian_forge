from glob import glob
from distutils.util import get_platform
from distutils.util import convert_path, subst_vars
from distutils.errors import (
from distutils import log, dir_util
from distutils.command.build_scripts import first_line_re
from distutils.spawn import find_executable
from distutils.command import install
import sys
import os
from typing import Dict, List
import zipimport
import shutil
import tempfile
import zipfile
import re
import stat
import random
import textwrap
import warnings
import site
import struct
import contextlib
import subprocess
import shlex
import io
import configparser
import sysconfig
from sysconfig import get_path
from setuptools import Command
from setuptools.sandbox import run_setup
from setuptools.command import setopt
from setuptools.archive_util import unpack_archive
from setuptools.package_index import (
from setuptools.command import bdist_egg, egg_info
from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from setuptools.wheel import Wheel
from pkg_resources import (
import pkg_resources
from ..compat import py39, py311
from .._path import ensure_directory
from ..extern.jaraco.text import yield_lines
class ScriptWriter:
    """
    Encapsulates behavior around writing entry point scripts for console and
    gui apps.
    """
    template = textwrap.dedent("\n        # EASY-INSTALL-ENTRY-SCRIPT: %(spec)r,%(group)r,%(name)r\n        import re\n        import sys\n\n        # for compatibility with easy_install; see #2198\n        __requires__ = %(spec)r\n\n        try:\n            from importlib.metadata import distribution\n        except ImportError:\n            try:\n                from importlib_metadata import distribution\n            except ImportError:\n                from pkg_resources import load_entry_point\n\n\n        def importlib_load_entry_point(spec, group, name):\n            dist_name, _, _ = spec.partition('==')\n            matches = (\n                entry_point\n                for entry_point in distribution(dist_name).entry_points\n                if entry_point.group == group and entry_point.name == name\n            )\n            return next(matches).load()\n\n\n        globals().setdefault('load_entry_point', importlib_load_entry_point)\n\n\n        if __name__ == '__main__':\n            sys.argv[0] = re.sub(r'(-script\\.pyw?|\\.exe)?$', '', sys.argv[0])\n            sys.exit(load_entry_point(%(spec)r, %(group)r, %(name)r)())\n        ").lstrip()
    command_spec_class = CommandSpec

    @classmethod
    def get_args(cls, dist, header=None):
        """
        Yield write_script() argument tuples for a distribution's
        console_scripts and gui_scripts entry points.
        """
        if header is None:
            header = cls.get_header()
        spec = str(dist.as_requirement())
        for type_ in ('console', 'gui'):
            group = type_ + '_scripts'
            for name, ep in dist.get_entry_map(group).items():
                cls._ensure_safe_name(name)
                script_text = cls.template % locals()
                args = cls._get_script_args(type_, name, header, script_text)
                yield from args

    @staticmethod
    def _ensure_safe_name(name):
        """
        Prevent paths in *_scripts entry point names.
        """
        has_path_sep = re.search('[\\\\/]', name)
        if has_path_sep:
            raise ValueError('Path separators not allowed in script names')

    @classmethod
    def best(cls):
        """
        Select the best ScriptWriter for this environment.
        """
        if sys.platform == 'win32' or (os.name == 'java' and os._name == 'nt'):
            return WindowsScriptWriter.best()
        else:
            return cls

    @classmethod
    def _get_script_args(cls, type_, name, header, script_text):
        yield (name, header + script_text)

    @classmethod
    def get_header(cls, script_text='', executable=None):
        """Create a #! line, getting options (if any) from script_text"""
        cmd = cls.command_spec_class.best().from_param(executable)
        cmd.install_options(script_text)
        return cmd.as_header()