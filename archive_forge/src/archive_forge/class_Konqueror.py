import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
class Konqueror(BaseBrowser):
    """Controller for the KDE File Manager (kfm, or Konqueror).

    See the output of ``kfmclient --commands``
    for more information on the Konqueror remote-control interface.
    """

    def open(self, url, new=0, autoraise=True):
        sys.audit('webbrowser.open', url)
        if new == 2:
            action = 'newTab'
        else:
            action = 'openURL'
        devnull = subprocess.DEVNULL
        try:
            p = subprocess.Popen(['kfmclient', action, url], close_fds=True, stdin=devnull, stdout=devnull, stderr=devnull)
        except OSError:
            pass
        else:
            p.wait()
            return True
        try:
            p = subprocess.Popen(['konqueror', '--silent', url], close_fds=True, stdin=devnull, stdout=devnull, stderr=devnull, start_new_session=True)
        except OSError:
            pass
        else:
            if p.poll() is None:
                return True
        try:
            p = subprocess.Popen(['kfm', '-d', url], close_fds=True, stdin=devnull, stdout=devnull, stderr=devnull, start_new_session=True)
        except OSError:
            return False
        else:
            return p.poll() is None