import os
import sys
import tempfile
import operator
import functools
import itertools
import re
import contextlib
import pickle
import textwrap
import builtins
import pkg_resources
from distutils.errors import DistutilsError
from pkg_resources import working_set
class SandboxViolation(DistutilsError):
    """A setup script attempted to modify the filesystem outside the sandbox"""
    tmpl = textwrap.dedent("\n        SandboxViolation: {cmd}{args!r} {kwargs}\n\n        The package setup script has attempted to modify files on your system\n        that are not within the EasyInstall build area, and has been aborted.\n\n        This package cannot be safely installed by EasyInstall, and may not\n        support alternate installation locations even if you run its setup\n        script by hand.  Please inform the package's author and the EasyInstall\n        maintainers to find out if a fix or workaround is available.\n        ").lstrip()

    def __str__(self):
        cmd, args, kwargs = self.args
        return self.tmpl.format(**locals())