from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
Add information about an edge linking two critical points.

        This actually describes two edges:

        from_idx ------ idx ------ to_idx

        However, in practice, from_idx and to_idx will typically be
        atom nuclei, with the center node (idx) referring to a bond
        critical point. Thus, it will be more convenient to model
        this as a single edge linking nuclei with the properties
        of the bond critical point stored as an edge attribute.

        Args:
            idx: index of node
            from_idx: from index of node
            from_lvec: vector of lattice image the from node is in
                as tuple of ints
            to_idx: to index of node
            to_lvec: vector of lattice image the to node is in as
                tuple of ints
        