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
def _find_gst_binaries():
    """Returns a list of GStreamer plugins and libraries to pass as the
    ``binaries`` argument of ``Analysis``.
    """
    gst_plugin_path = _find_gst_plugin_path()
    plugin_filepaths = []
    for plugin_dir in gst_plugin_path:
        plugin_filepaths.extend(glob.glob(os.path.join(plugin_dir, 'libgst*')))
    if len(plugin_filepaths) == 0:
        logging.warning('Could not find GStreamer plugins. ' + 'Possible solution: set GST_PLUGIN_PATH')
        return []
    lib_filepaths = set()
    for plugin_filepath in plugin_filepaths:
        plugin_deps = bindepend.selectImports(plugin_filepath)
        lib_filepaths.update([path for _, path in plugin_deps])
    plugin_binaries = [(f, 'gst-plugins') for f in plugin_filepaths]
    lib_binaries = [(f, '.') for f in lib_filepaths]
    return plugin_binaries + lib_binaries