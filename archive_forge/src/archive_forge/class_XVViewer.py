from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
class XVViewer(UnixViewer):
    """
    The X Viewer ``xv`` command.
    This viewer supports the ``title`` parameter.
    """

    def get_command_ex(self, file, title=None, **options):
        command = executable = 'xv'
        if title:
            command += f' -name {quote(title)}'
        return (command, executable)

    def show_file(self, path, **options):
        """
        Display given file.
        """
        args = ['xv']
        title = options.get('title')
        if title:
            args += ['-name', title]
        args.append(path)
        subprocess.Popen(args)
        return 1