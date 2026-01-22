from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
class XDGViewer(UnixViewer):
    """
    The freedesktop.org ``xdg-open`` command.
    """

    def get_command_ex(self, file, **options):
        command = executable = 'xdg-open'
        return (command, executable)

    def show_file(self, path, **options):
        """
        Display given file.
        """
        subprocess.Popen(['xdg-open', path])
        return 1