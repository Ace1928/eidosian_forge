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
def as_tmpfile(self, tmpdir=None):
    """
        Copy the pseudopotential to a temporary a file and returns a new pseudopotential object.
        Useful for unit tests in which we have to change the content of the file.

        Args:
            tmpdir: If None, a new temporary directory is created and files are copied here
                else tmpdir is used.
        """
    tmpdir = tempfile.mkdtemp() if tmpdir is None else tmpdir
    new_path = os.path.join(tmpdir, self.basename)
    shutil.copy(self.filepath, new_path)
    root, _ext = os.path.splitext(self.filepath)
    dj_report = root + '.djrepo'
    if os.path.isfile(dj_report):
        shutil.copy(dj_report, os.path.join(tmpdir, os.path.basename(dj_report)))
    new = type(self).from_file(new_path)
    if self.has_dojo_report:
        new.dojo_report = self.dojo_report.deepcopy()
    return new