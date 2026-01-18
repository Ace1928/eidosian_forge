from __future__ import annotations
from collections import namedtuple
from typing import TYPE_CHECKING
from monty.functools import lazy_property
from monty.json import MSONable
from pymatgen.core.libxcfunc import LibxcFunc
The name of the functional. If the functional is not found in the aliases,
        the string has the form X_NAME+C_NAME.
        