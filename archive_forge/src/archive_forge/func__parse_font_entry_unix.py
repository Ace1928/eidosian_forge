import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def _parse_font_entry_unix(entry, fonts):
    """
    Parses an entry in the unix font data to add to the pygame font
    dictionary.

    :param entry: A entry from the unix font list.
    :param fonts: The pygame font dictionary to add the parsed font data to.

    """
    filename, family, style = entry.split(':', 2)
    if splitext(filename)[1].lower() in OpenType_extensions:
        bold = 'Bold' in style
        italic = 'Italic' in style
        oblique = 'Oblique' in style
        for name in family.strip().split(','):
            if name:
                break
        else:
            name = splitext(basename(filename))[0]
        _addfont(_simplename(name), bold, italic or oblique, filename, fonts)