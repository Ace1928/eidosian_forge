from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.core import Element, Molecule

    Create an OpenFF Molecule from a SMILES string and optional geometry.

    Constructs an OpenFF Molecule from the provided SMILES
    string, adds conformers based on the provided geometry (if
    any), assigns partial charges using the specified method
    or provided partial charges, and applies charge scaling.

    Args:
        smile (str): The SMILES string of the molecule.
        geometry (Union[pymatgen.core.Molecule, str, Path, None], optional): The
            geometry to use for adding conformers. Can be a Pymatgen Molecule,
            file path, or None.
        charge_scaling (float, optional): The scaling factor for partial charges.
            Default is 1.
        partial_charges (Union[List[float], None], optional): A list of partial
            charges to assign, or None to use the charge method.
        backup_charge_method (str, optional): The backup charge method to use if
            partial charges are not provided. Default is "am1bcc".

    Returns:
        tk.Molecule: The created OpenFF Molecule.
    