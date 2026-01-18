from __future__ import annotations
import re
import numpy as np
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
from pymatgen.analysis.chemenv.utils.chemenv_errors import NeighborsNotComputedChemenvError
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import rotateCoords
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Molecule
from pymatgen.io.cif import CifParser

    Compute the environments.

    Args:
        chemenv_configuration:
    