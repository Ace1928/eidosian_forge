import os, time
import re
from xdg.IniFile import IniFile, is_ascii
from xdg.BaseDirectory import xdg_data_dirs
from xdg.Exceptions import NoThemeError, debug
import xdg.Config
def __get_themes(themename):
    """Generator yielding IconTheme objects for a specified theme and any themes
    from which it inherits.
    """
    for dir in icondirs:
        theme_file = os.path.join(dir, themename, 'index.theme')
        if os.path.isfile(theme_file):
            break
        theme_file = os.path.join(dir, themename, 'index.desktop')
        if os.path.isfile(theme_file):
            break
    else:
        if debug:
            raise NoThemeError(themename)
        return
    theme = IconTheme()
    theme.parse(theme_file)
    yield theme
    for subtheme in theme.getInherits():
        for t in __get_themes(subtheme):
            yield t