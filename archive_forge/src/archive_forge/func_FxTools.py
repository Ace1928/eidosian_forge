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
def FxTools(self):
    """
        Microsoft .NET Framework Tools.

        Return
        ------
        list of str
            paths
        """
    pi = self.pi
    si = self.si
    if self.vs_ver <= 10.0:
        include32 = True
        include64 = not pi.target_is_x86() and (not pi.current_is_x86())
    else:
        include32 = pi.target_is_x86() or pi.current_is_x86()
        include64 = pi.current_cpu == 'amd64' or pi.target_cpu == 'amd64'
    tools = []
    if include32:
        tools += [join(si.FrameworkDir32, ver) for ver in si.FrameworkVersion32]
    if include64:
        tools += [join(si.FrameworkDir64, ver) for ver in si.FrameworkVersion64]
    return tools