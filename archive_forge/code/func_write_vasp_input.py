from __future__ import annotations
import os
import re
from multiprocessing import Pool
from typing import TYPE_CHECKING, Callable
from pymatgen.alchemy.materials import TransformedStructure
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet
def write_vasp_input(self, **kwargs):
    """Batch write vasp input for a sequence of transformed structures to
        output_dir, following the format output_dir/{formula}_{number}.

        Args:
            kwargs: All kwargs supported by batch_write_vasp_input.
        """
    batch_write_vasp_input(self.transformed_structures, **kwargs)