from xdg.Menu import parse, Menu, MenuEntry
import os
import locale
import subprocess
import ast
import sys
from xdg.BaseDirectory import xdg_data_dirs, xdg_config_dirs
from xdg.DesktopEntry import DesktopEntry
from xdg.Exceptions import ParsingError
from xdg.util import PY3
import xdg.Locale
import xdg.Config
def __addFiles(self, dir_, subdir, prefix, legacy):
    for item in os.listdir(os.path.join(dir_, subdir)):
        if item.endswith('.desktop'):
            try:
                menuentry = MenuEntry(os.path.join(subdir, item), dir_, prefix)
            except ParsingError:
                continue
            self.cacheEntries[dir_].append(menuentry)
            if legacy:
                self.cacheEntries['legacy'].append(menuentry)
        elif os.path.isdir(os.path.join(dir_, subdir, item)) and (not legacy):
            self.__addFiles(dir_, os.path.join(subdir, item), prefix, legacy)