from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
def _get_terminal_display_formatter(self, spacer='  '):
    """ generate function to use for terminal formatting
        """
    dirname_output_format = '%s/'
    fname_output_format = spacer + '%s'
    fp_format = '%s/%s'
    return self._get_display_formatter(dirname_output_format, fname_output_format, fp_format)