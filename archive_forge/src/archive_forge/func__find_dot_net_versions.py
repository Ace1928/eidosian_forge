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
def _find_dot_net_versions(self, bits):
    """
        Find Microsoft .NET Framework versions.

        Parameters
        ----------
        bits: int
            Platform number of bits: 32 or 64.

        Return
        ------
        tuple of str
            versions
        """
    reg_ver = self.ri.lookup(self.ri.vc, 'frameworkver%d' % bits)
    dot_net_dir = getattr(self, 'FrameworkDir%d' % bits)
    ver = reg_ver or self._use_last_dir_name(dot_net_dir, 'v') or ''
    if self.vs_ver >= 12.0:
        return (ver, 'v4.0')
    elif self.vs_ver >= 10.0:
        return ('v4.0.30319' if ver.lower()[:2] != 'v4' else ver, 'v3.5')
    elif self.vs_ver == 9.0:
        return ('v3.5', 'v2.0.50727')
    elif self.vs_ver == 8.0:
        return ('v3.0', 'v2.0.50727')