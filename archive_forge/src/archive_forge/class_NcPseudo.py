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
class NcPseudo(abc.ABC):
    """
    Abstract class defining the methods that must be implemented
    by the concrete classes representing norm-conserving pseudopotentials.
    """

    @property
    @abc.abstractmethod
    def nlcc_radius(self):
        """
        Radius at which the core charge vanish (i.e. cut-off in a.u.).
        Returns 0.0 if nlcc is not used.
        """

    @property
    def has_nlcc(self):
        """True if the pseudo is generated with non-linear core correction."""
        return self.nlcc_radius > 0.0

    @property
    def rcore(self):
        """Radius of the pseudization sphere in a.u."""
        try:
            return self._core
        except AttributeError:
            return None