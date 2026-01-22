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
@property
def VCTools(self):
    """
        Microsoft Visual C++ Tools.

        Return
        ------
        list of str
            paths
        """
    si = self.si
    tools = [join(si.VCInstallDir, 'VCPackages')]
    forcex86 = True if self.vs_ver <= 10.0 else False
    arch_subdir = self.pi.cross_dir(forcex86)
    if arch_subdir:
        tools += [join(si.VCInstallDir, 'Bin%s' % arch_subdir)]
    if self.vs_ver == 14.0:
        path = 'Bin%s' % self.pi.current_dir(hidex86=True)
        tools += [join(si.VCInstallDir, path)]
    elif self.vs_ver >= 15.0:
        host_dir = 'bin\\HostX86%s' if self.pi.current_is_x86() else 'bin\\HostX64%s'
        tools += [join(si.VCInstallDir, host_dir % self.pi.target_dir(x64=True))]
        if self.pi.current_cpu != self.pi.target_cpu:
            tools += [join(si.VCInstallDir, host_dir % self.pi.current_dir(x64=True))]
    else:
        tools += [join(si.VCInstallDir, 'Bin')]
    return tools