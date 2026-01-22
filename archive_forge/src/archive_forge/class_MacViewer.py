from __future__ import annotations
import os
import shutil
import subprocess
import sys
from shlex import quote
from . import Image
class MacViewer(Viewer):
    """The default viewer on macOS using ``Preview.app``."""
    format = 'PNG'
    options = {'compress_level': 1, 'save_all': True}

    def get_command(self, file, **options):
        command = 'open -a Preview.app'
        command = f'({command} {quote(file)}; sleep 20; rm -f {quote(file)})&'
        return command

    def show_file(self, path, **options):
        """
        Display given file.
        """
        subprocess.call(['open', '-a', 'Preview.app', path])
        executable = sys.executable or shutil.which('python3')
        if executable:
            subprocess.Popen([executable, '-c', 'import os, sys, time; time.sleep(20); os.remove(sys.argv[1])', path])
        return 1