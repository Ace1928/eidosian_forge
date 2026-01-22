import collections
from functools import reduce, singledispatch
from typing import (Any, Dict, Iterable, List, Optional,
import numpy as np
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData, Info
from ase.utils.plotting import SimplePlottingAxes
Get relevent dict entries in tuple form

            e.g. if data.info = {'a': 1, 'b': 2, 'c': 3}
                 and info_keys = ('a', 'c')

                 then return (('a', 1), ('c': 3))
            