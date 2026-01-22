from __future__ import annotations
import errno
import json
import os
import platform
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any
from jupyter_client.kernelspec import KernelSpecManager
from traitlets import Unicode
from traitlets.config import Application
class InstallIPythonKernelSpecApp(Application):
    """Dummy app wrapping argparse"""
    name = Unicode('ipython-kernel-install')

    def initialize(self, argv: list[str] | None=None) -> None:
        """Initialize the app."""
        if argv is None:
            argv = sys.argv[1:]
        self.argv = argv

    def start(self) -> None:
        """Start the app."""
        import argparse
        parser = argparse.ArgumentParser(prog=self.name, description='Install the IPython kernel spec.')
        parser.add_argument('--user', action='store_true', help='Install for the current user instead of system-wide')
        parser.add_argument('--name', type=str, default=KERNEL_NAME, help='Specify a name for the kernelspec. This is needed to have multiple IPython kernels at the same time.')
        parser.add_argument('--display-name', type=str, help='Specify the display name for the kernelspec. This is helpful when you have multiple IPython kernels.')
        parser.add_argument('--profile', type=str, help='Specify an IPython profile to load. This can be used to create custom versions of the kernel.')
        parser.add_argument('--prefix', type=str, help='Specify an install prefix for the kernelspec. This is needed to install into a non-default location, such as a conda/virtual-env.')
        parser.add_argument('--sys-prefix', action='store_const', const=sys.prefix, dest='prefix', help="Install to Python's sys.prefix. Shorthand for --prefix='%s'. For use in conda/virtual-envs." % sys.prefix)
        parser.add_argument('--env', action='append', nargs=2, metavar=('ENV', 'VALUE'), help='Set environment variables for the kernel.')
        parser.add_argument('--frozen_modules', action='store_true', help='Enable frozen modules for potentially faster startup. This has a downside of preventing the debugger from navigating to certain built-in modules.')
        opts = parser.parse_args(self.argv)
        if opts.env:
            opts.env = dict(opts.env)
        try:
            dest = install(user=opts.user, kernel_name=opts.name, profile=opts.profile, prefix=opts.prefix, display_name=opts.display_name, env=opts.env)
        except OSError as e:
            if e.errno == errno.EACCES:
                print(e, file=sys.stderr)
                if opts.user:
                    print('Perhaps you want `sudo` or `--user`?', file=sys.stderr)
                self.exit(1)
            raise
        print(f'Installed kernelspec {opts.name} in {dest}')