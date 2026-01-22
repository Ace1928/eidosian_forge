import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
from distutils.errors import (
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
import threading
def CCompiler_get_version(self, force=False, ok_status=[0]):
    """
    Return compiler version, or None if compiler is not available.

    Parameters
    ----------
    force : bool, optional
        If True, force a new determination of the version, even if the
        compiler already has a version attribute. Default is False.
    ok_status : list of int, optional
        The list of status values returned by the version look-up process
        for which a version string is returned. If the status value is not
        in `ok_status`, None is returned. Default is ``[0]``.

    Returns
    -------
    version : str or None
        Version string, in the format of `distutils.version.LooseVersion`.

    """
    if not force and hasattr(self, 'version'):
        return self.version
    self.find_executables()
    try:
        version_cmd = self.version_cmd
    except AttributeError:
        return None
    if not version_cmd or not version_cmd[0]:
        return None
    try:
        matcher = self.version_match
    except AttributeError:
        try:
            pat = self.version_pattern
        except AttributeError:
            return None

        def matcher(version_string):
            m = re.match(pat, version_string)
            if not m:
                return None
            version = m.group('version')
            return version
    try:
        output = subprocess.check_output(version_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        output = exc.output
        status = exc.returncode
    except OSError:
        status = 127
        output = b''
    else:
        output = filepath_from_subprocess_output(output)
        status = 0
    version = None
    if status in ok_status:
        version = matcher(output)
        if version:
            version = LooseVersion(version)
    self.version = version
    return version