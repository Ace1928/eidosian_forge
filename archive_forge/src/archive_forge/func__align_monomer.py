from __future__ import annotations
import os
import tempfile
from shutil import which
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.tempfile import ScratchDir
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.util.coord import get_angle
def _align_monomer(self, monomer: Molecule, mon_vector: ArrayLike, move_direction: ArrayLike) -> None:
    """
        rotate the monomer so that it is aligned along the move direction.

        Args:
            monomer (Molecule)
            mon_vector (numpy.array): molecule vector that starts from the
                start atom index to the end atom index
            move_direction (numpy.array): the direction of the polymer chain
                extension
        """
    axis = np.cross(mon_vector, move_direction)
    origin = monomer[self.start].coords
    angle = get_angle(mon_vector, move_direction)
    op = SymmOp.from_origin_axis_angle(origin, axis, angle)
    monomer.apply_operation(op)