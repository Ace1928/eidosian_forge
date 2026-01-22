from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.core import Element, Molecule
def add_conformer(openff_mol: tk.Molecule, geometry: pymatgen.core.Molecule | None) -> tuple[tk.Molecule, dict[int, int]]:
    """
    Add conformers to an OpenFF Molecule based on the provided geometry.

    If a geometry is provided, infers an OpenFF Molecule from it,
    finds an atom mapping between the inferred molecule and the
    input molecule, and adds the conformer coordinates to the input
    molecule. If no geometry is provided, generates a single conformer.

    Args:
        openff_mol (tk.Molecule): The OpenFF Molecule to add conformers to.
        geometry (Union[pymatgen.core.Molecule, None]): The geometry to use for adding
            conformers.

    Returns:
        Tuple[tk.Molecule, Dict[int, int]]: A tuple containing the updated OpenFF
            Molecule with added conformers and a dictionary representing the atom
            mapping.
    """
    if geometry:
        inferred_mol = infer_openff_mol(geometry)
        is_isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
        if not is_isomorphic:
            raise ValueError(f'An isomorphism cannot be found between smile {openff_mol.to_smiles()}and the provided molecule {geometry}.')
        new_mol = pymatgen.core.Molecule.from_sites([geometry.sites[i] for i in atom_map.values()])
        openff_mol.add_conformer(new_mol.cart_coords * unit.angstrom)
    else:
        atom_map = {i: i for i in range(openff_mol.n_atoms)}
        openff_mol.generate_conformers(n_conformers=1)
    return (openff_mol, atom_map)