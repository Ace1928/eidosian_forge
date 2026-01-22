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
def _get_menu_file_path(filename):
    dirs = list(xdg_config_dirs)
    if xdg.Config.root_mode is True:
        dirs.pop(0)
    for d in dirs:
        menuname = os.path.join(d, 'menus', filename)
        if os.path.isfile(menuname):
            return menuname