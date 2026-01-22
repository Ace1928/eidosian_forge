import json
from os import listdir, pathsep
from os.path import join, isfile, isdir, dirname
from subprocess import CalledProcessError
import contextlib
import platform
import itertools
import subprocess
import distutils.errors
from setuptools.extern.more_itertools import unique_everseen
def _guess_vc_legacy(self):
    """
        Locate Visual C++ for versions prior to 2017.

        Return
        ------
        str
            path
        """
    default = join(self.ProgramFilesx86, 'Microsoft Visual Studio %0.1f\\VC' % self.vs_ver)
    reg_path = join(self.ri.vc_for_python, '%0.1f' % self.vs_ver)
    python_vc = self.ri.lookup(reg_path, 'installdir')
    default_vc = join(python_vc, 'VC') if python_vc else default
    return self.ri.lookup(self.ri.vc, '%0.1f' % self.vs_ver) or default_vc