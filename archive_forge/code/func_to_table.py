from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
def to_table(self, filter_function=None):
    """Return string with data in tabular form."""
    table = []
    for p in self:
        if filter_function is not None and filter_function(p):
            continue
        table.append([p.basename, p.symbol, p.Z_val, p.l_max, p.l_local, p.xc, p.type])
    return tabulate(table, headers=['basename', 'symbol', 'Z_val', 'l_max', 'l_local', 'XC', 'type'], tablefmt='grid')