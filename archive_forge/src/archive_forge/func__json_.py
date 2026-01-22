import re
import sys
from html import escape
from weakref import proxy
from .std import tqdm as std_tqdm
def _json_(self, pretty=None):
    pbar = getattr(self, 'pbar', None)
    if pbar is None:
        return {}
    d = pbar.format_dict
    if pretty is not None:
        d['ascii'] = not pretty
    return d