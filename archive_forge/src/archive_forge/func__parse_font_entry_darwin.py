import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def _parse_font_entry_darwin(name, filepath, fonts):
    """
    Parses a font entry for macOS

    :param name: The filepath without extensions or directories
    :param filepath: The full path to the font
    :param fonts: The pygame font dictionary to add the parsed font data to.
    """
    name = _simplename(name)
    mods = ('regular',)
    for mod in mods:
        if mod in name:
            name = name.replace(mod, '')
    bold = italic = False
    if 'bold' in name:
        name = name.replace('bold', '')
        bold = True
    if 'italic' in name:
        name = name.replace('italic', '')
        italic = True
    _addfont(name, bold, italic, filepath, fonts)