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
class AbinitPseudo(Pseudo):
    """An AbinitPseudo is a pseudopotential whose file contains an abinit header."""

    def __init__(self, path, header):
        """
        Args:
            path: Filename.
            header: AbinitHeader instance.
        """
        self.path = path
        self.header = header
        self._summary = header.summary
        self.xc = XcFunc.from_abinit_ixc(header['pspxc'])
        for attr_name in header:
            value = header.get(attr_name)
            setattr(self, '_' + attr_name, value)

    @property
    def summary(self):
        """Summary line reported in the ABINIT header."""
        return self._summary.strip()

    @property
    def Z(self):
        return self._zatom

    @property
    def Z_val(self):
        return self._zion

    @property
    def l_max(self):
        return self._lmax

    @property
    def l_local(self):
        return self._lloc

    @property
    def supports_soc(self):
        if self._pspcod == 8:
            switch = self.header['extension_switch']
            if switch in (0, 1):
                return False
            if switch in (2, 3):
                return True
            raise ValueError(f"Don't know how to handle extension_switch: {switch}")
        return False