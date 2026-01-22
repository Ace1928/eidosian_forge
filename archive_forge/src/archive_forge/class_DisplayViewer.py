from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
class DisplayViewer(UnixViewer):
    """
    The ImageMagick ``display`` command.
    This viewer supports the ``title`` parameter.
    """

    def get_command_ex(self, file, title=None, **options):
        command = executable = 'display'
        if title:
            command += f' -title {quote(title)}'
        return (command, executable)

    def show_file(self, path, **options):
        """
        Display given file.
        """
        args = ['display']
        title = options.get('title')
        if title:
            args += ['-title', title]
        args.append(path)
        subprocess.Popen(args)
        return 1