imported modules that pyinstaller would not find on its own using
import os
import sys
import pkgutil
import logging
from os.path import dirname, join
import importlib
import subprocess
import re
import glob
import kivy
from kivy.factory import Factory
from PyInstaller.depend import bindepend
from os import environ
def _find_gst_plugin_path():
    """Returns a list of directories to search for GStreamer plugins.
    """
    if 'GST_PLUGIN_PATH' in environ:
        return [os.path.abspath(os.path.expanduser(path)) for path in environ['GST_PLUGIN_PATH'].split(os.pathsep)]
    try:
        p = subprocess.Popen(['gst-inspect-1.0', 'coreelements'], stdout=subprocess.PIPE, universal_newlines=True)
    except:
        return []
    stdoutdata, stderrdata = p.communicate()
    match = re.search('\\s+(\\S+libgstcoreelements\\.\\S+)', stdoutdata)
    if not match:
        return []
    return [os.path.dirname(match.group(1))]