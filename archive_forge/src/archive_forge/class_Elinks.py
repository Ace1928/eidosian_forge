import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
class Elinks(UnixBrowser):
    """Launcher class for Elinks browsers."""
    remote_args = ['-remote', 'openURL(%s%action)']
    remote_action = ''
    remote_action_newwin = ',new-window'
    remote_action_newtab = ',new-tab'
    background = False
    redirect_stdout = False