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
class PlatformInfo:
    """
    Current and Target Architectures information.

    Parameters
    ----------
    arch: str
        Target architecture.
    """
    current_cpu = environ.get('processor_architecture', '').lower()

    def __init__(self, arch):
        self.arch = arch.lower().replace('x64', 'amd64')

    @property
    def target_cpu(self):
        """
        Return Target CPU architecture.

        Return
        ------
        str
            Target CPU
        """
        return self.arch[self.arch.find('_') + 1:]

    def target_is_x86(self):
        """
        Return True if target CPU is x86 32 bits..

        Return
        ------
        bool
            CPU is x86 32 bits
        """
        return self.target_cpu == 'x86'

    def current_is_x86(self):
        """
        Return True if current CPU is x86 32 bits..

        Return
        ------
        bool
            CPU is x86 32 bits
        """
        return self.current_cpu == 'x86'

    def current_dir(self, hidex86=False, x64=False):
        """
        Current platform specific subfolder.

        Parameters
        ----------
        hidex86: bool
            return '' and not '\x86' if architecture is x86.
        x64: bool
            return 'd' and not '\x07md64' if architecture is amd64.

        Return
        ------
        str
            subfolder: '	arget', or '' (see hidex86 parameter)
        """
        return '' if self.current_cpu == 'x86' and hidex86 else '\\x64' if self.current_cpu == 'amd64' and x64 else '\\%s' % self.current_cpu

    def target_dir(self, hidex86=False, x64=False):
        """
        Target platform specific subfolder.

        Parameters
        ----------
        hidex86: bool
            return '' and not '\\x86' if architecture is x86.
        x64: bool
            return '\\x64' and not '\\amd64' if architecture is amd64.

        Return
        ------
        str
            subfolder: '\\current', or '' (see hidex86 parameter)
        """
        return '' if self.target_cpu == 'x86' and hidex86 else '\\x64' if self.target_cpu == 'amd64' and x64 else '\\%s' % self.target_cpu

    def cross_dir(self, forcex86=False):
        """
        Cross platform specific subfolder.

        Parameters
        ----------
        forcex86: bool
            Use 'x86' as current architecture even if current architecture is
            not x86.

        Return
        ------
        str
            subfolder: '' if target architecture is current architecture,
            '\\current_target' if not.
        """
        current = 'x86' if forcex86 else self.current_cpu
        return '' if self.target_cpu == current else self.target_dir().replace('\\', '\\%s_' % current)