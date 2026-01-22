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
def add_dep_paths():
    """Should be called by the hook. It adds the paths with the binary
    dependencies to the system path so that pyinstaller can find the binaries
    during its crawling stage.
    """
    paths = []
    if old_deps is not None:
        for importer, modname, ispkg in pkgutil.iter_modules(old_deps.__path__):
            if not ispkg:
                continue
            try:
                module_spec = importer.find_spec(modname)
                mod = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(mod)
            except ImportError as e:
                logging.warning(f'deps: Error importing dependency: {e}')
                continue
            if hasattr(mod, 'dep_bins'):
                paths.extend(mod.dep_bins)
    sys.path.extend(paths)
    if kivy_deps is None:
        return
    paths = []
    for importer, modname, ispkg in pkgutil.iter_modules(kivy_deps.__path__):
        if not ispkg:
            continue
        try:
            module_spec = importer.find_spec(modname)
            mod = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(mod)
        except ImportError as e:
            logging.warning(f'deps: Error importing dependency: {e}')
            continue
        if hasattr(mod, 'dep_bins'):
            paths.extend(mod.dep_bins)
    sys.path.extend(paths)