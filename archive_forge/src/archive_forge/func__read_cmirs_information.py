from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
def _read_cmirs_information(self):
    """
        Parses information from CMIRS solvent calculations.

        In addition to the 5 energies returned by ISOSVP (and read separately in
        _read_isosvp_information), there are 4 additional energies reported, as shown
        in the example below

        --------------------------------------------------------------------------------
        The Final SS(V)PE energies and Properties
        --------------------------------------------------------------------------------

        Energies
        --------------------
        The Final Solution-Phase Energy =     -40.4751881546
        The Solute Internal Energy =          -40.4748568841
        The Change in Solute Internal Energy =  0.0000089729  (   0.00563 KCAL/MOL)
        The Reaction Field Free Energy =       -0.0003312705  (  -0.20788 KCAL/MOL)
        The Dispersion Energy =                 0.6955550107  (  -2.27836 KCAL/MOL)
        The Exchange Energy =                   0.2652679507  (   2.15397 KCAL/MOL)
        Min. Negative Field Energy =            0.0005235850  (   0.00000 KCAL/MOL)
        Max. Positive Field Energy =            0.0179866718  (   0.00000 KCAL/MOL)
        The Total Solvation Free Energy =      -0.0005205275  (  -0.32664 KCAL/MOL)
        """
    temp_dict = read_pattern(self.text, {'dispersion_e': '\\s*The Dispersion Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'exchange_e': '\\s*The Exchange Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'min_neg_field_e': '\\s*Min. Negative Field Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'max_pos_field_e': '\\s*Max. Positive Field Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*'})
    for key in temp_dict:
        if temp_dict.get(key) is None:
            self.data['solvent_data']['cmirs'][key] = None
        elif len(temp_dict.get(key)) == 1:
            self.data['solvent_data']['cmirs'][key] = float(temp_dict.get(key)[0][0])
        else:
            temp_result = np.zeros(len(temp_dict.get(key)))
            for ii, entry in enumerate(temp_dict.get(key)):
                temp_result[ii] = float(entry[0])
            self.data['solvent_data']['cmirs'][key] = temp_result