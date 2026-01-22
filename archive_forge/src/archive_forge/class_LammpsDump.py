from __future__ import annotations
import re
from glob import glob
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.lammps.data import LammpsBox
class LammpsDump(MSONable):
    """Object for representing dump data for a single snapshot."""

    def __init__(self, timestep: int, natoms: int, box: LammpsBox, data: pd.DataFrame) -> None:
        """
        Base constructor.

        Args:
            timestep (int): Current time step.
            natoms (int): Total number of atoms in the box.
            box (LammpsBox): Simulation box.
            data (pd.DataFrame): Dumped atomic data.
        """
        self.timestep = timestep
        self.natoms = natoms
        self.box = box
        self.data = data

    @classmethod
    def from_str(cls, string: str) -> Self:
        """
        Constructor from string parsing.

        Args:
            string (str): Input string.
        """
        lines = string.split('\n')
        time_step = int(lines[1])
        n_atoms = int(lines[3])
        box_arr = np.loadtxt(StringIO('\n'.join(lines[5:8])))
        bounds = box_arr[:, :2]
        tilt = None
        if 'xy xz yz' in lines[4]:
            tilt = box_arr[:, 2]
            x = (0, tilt[0], tilt[1], tilt[0] + tilt[1])
            y = (0, tilt[2])
            bounds -= np.array([[min(x), max(x)], [min(y), max(y)], [0, 0]])
        box = LammpsBox(bounds, tilt)
        data_head = lines[8].replace('ITEM: ATOMS', '').split()
        data = pd.read_csv(StringIO('\n'.join(lines[9:])), names=data_head, delim_whitespace=True)
        return cls(time_step, n_atoms, box, data)

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): Dict representation.

        Returns:
            LammpsDump
        """
        items = {'timestep': dct['timestep'], 'natoms': dct['natoms']}
        items['box'] = LammpsBox.from_dict(dct['box'])
        items['data'] = pd.read_json(dct['data'], orient='split')
        return cls(**items)

    def as_dict(self) -> dict[str, Any]:
        """Returns: MSONable dict."""
        dct: dict[str, Any] = {}
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        dct['timestep'] = self.timestep
        dct['natoms'] = self.natoms
        dct['box'] = self.box.as_dict()
        dct['data'] = self.data.to_json(orient='split')
        return dct