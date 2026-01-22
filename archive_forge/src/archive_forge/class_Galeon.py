import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
class Galeon(UnixBrowser):
    """Launcher class for Galeon/Epiphany browsers."""
    raise_opts = ['-noraise', '']
    remote_args = ['%action', '%s']
    remote_action = '-n'
    remote_action_newwin = '-w'
    background = True