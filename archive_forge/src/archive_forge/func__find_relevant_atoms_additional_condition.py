from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
def _find_relevant_atoms_additional_condition(self, isite, icohps, additional_condition):
    """
        Will find all relevant atoms that fulfill the additional_conditions.

        Args:
            isite: number of site in structure (starts with 0)
            icohps: icohps
            additional_condition (int): additional condition

        Returns:
            tuple: keys, lengths and neighbors from selected ICOHPs and selected ICOHPs
        """
    neighbors_from_ICOHPs = []
    lengths_from_ICOHPs = []
    icohps_from_ICOHPs = []
    keys_from_ICOHPs = []
    for key, icohp in icohps.items():
        atomnr1 = self._get_atomnumber(icohp._atom1)
        atomnr2 = self._get_atomnumber(icohp._atom2)
        if additional_condition in (1, 3, 5, 6):
            val1 = self.valences[atomnr1]
            val2 = self.valences[atomnr2]
        if additional_condition == 0:
            if atomnr1 == isite:
                neighbors_from_ICOHPs.append(atomnr2)
                lengths_from_ICOHPs.append(icohp._length)
                icohps_from_ICOHPs.append(icohp.summed_icohp)
                keys_from_ICOHPs.append(key)
            elif atomnr2 == isite:
                neighbors_from_ICOHPs.append(atomnr1)
                lengths_from_ICOHPs.append(icohp._length)
                icohps_from_ICOHPs.append(icohp.summed_icohp)
                keys_from_ICOHPs.append(key)
        elif additional_condition == 1:
            if val1 < 0.0 < val2 or val2 < 0.0 < val1:
                if atomnr1 == isite:
                    neighbors_from_ICOHPs.append(atomnr2)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
                elif atomnr2 == isite:
                    neighbors_from_ICOHPs.append(atomnr1)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
        elif additional_condition == 2:
            if icohp._atom1.rstrip('0123456789') != icohp._atom2.rstrip('0123456789'):
                if atomnr1 == isite:
                    neighbors_from_ICOHPs.append(atomnr2)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
                elif atomnr2 == isite:
                    neighbors_from_ICOHPs.append(atomnr1)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
        elif additional_condition == 3:
            if (val1 < 0.0 < val2 or val2 < 0.0 < val1) and icohp._atom1.rstrip('0123456789') != icohp._atom2.rstrip('0123456789'):
                if atomnr1 == isite:
                    neighbors_from_ICOHPs.append(atomnr2)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
                elif atomnr2 == isite:
                    neighbors_from_ICOHPs.append(atomnr1)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
        elif additional_condition == 4:
            if icohp._atom1.rstrip('0123456789') == 'O' or icohp._atom2.rstrip('0123456789') == 'O':
                if atomnr1 == isite:
                    neighbors_from_ICOHPs.append(atomnr2)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
                elif atomnr2 == isite:
                    neighbors_from_ICOHPs.append(atomnr1)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
        elif additional_condition == 5:
            if val1 > 0.0 and val2 > 0.0 or (val1 < 0.0 and val2 < 0.0):
                if atomnr1 == isite:
                    neighbors_from_ICOHPs.append(atomnr2)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
                elif atomnr2 == isite:
                    neighbors_from_ICOHPs.append(atomnr1)
                    lengths_from_ICOHPs.append(icohp._length)
                    icohps_from_ICOHPs.append(icohp.summed_icohp)
                    keys_from_ICOHPs.append(key)
        elif additional_condition == 6 and val1 > 0.0 and (val2 > 0.0):
            if atomnr1 == isite:
                neighbors_from_ICOHPs.append(atomnr2)
                lengths_from_ICOHPs.append(icohp._length)
                icohps_from_ICOHPs.append(icohp.summed_icohp)
                keys_from_ICOHPs.append(key)
            elif atomnr2 == isite:
                neighbors_from_ICOHPs.append(atomnr1)
                lengths_from_ICOHPs.append(icohp._length)
                icohps_from_ICOHPs.append(icohp.summed_icohp)
                keys_from_ICOHPs.append(key)
    return (keys_from_ICOHPs, lengths_from_ICOHPs, neighbors_from_ICOHPs, icohps_from_ICOHPs)