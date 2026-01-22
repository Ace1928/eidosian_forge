import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
class Chrome(UnixBrowser):
    """Launcher class for Google Chrome browser."""
    remote_args = ['%action', '%s']
    remote_action = ''
    remote_action_newwin = '--new-window'
    remote_action_newtab = ''
    background = True