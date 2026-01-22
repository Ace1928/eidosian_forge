from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
def _apply_transformation(inputs):
    """Helper method for multiprocessing of apply_transformation. Must not be
    in the class so that it can be pickled.

    Args:
        inputs: Tuple containing the transformed structure, the transformation
            to be applied, a boolean indicating whether to extend the
            collection, and a boolean indicating whether to clear the redo

    Returns:
        list[Structure]: the modified initial structure, plus
            any new structures created by a one-to-many transformation
    """
    ts, transformation, extend_collection, clear_redo = inputs
    new = ts.append_transformation(transformation, extend_collection, clear_redo=clear_redo)
    o = [ts]
    if new:
        o.extend(new)
    return o