from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode
from IPython.core.display import DisplayObject, TextDisplayObject
from typing import Tuple, Iterable, Optional
def _data_and_metadata(self):
    """shortcut for returning metadata with url information, if defined"""
    md = {}
    if self.url:
        md['url'] = self.url
    if md:
        return (self.data, md)
    else:
        return self.data