from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
def undo_last_change(self) -> None:
    """Undo the last transformation in the TransformedStructure.

        Raises:
            IndexError if already at the oldest change.
        """
    for x in self.transformed_structures:
        x.undo_last_change()