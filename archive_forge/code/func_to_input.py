from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as cst
from monty.io import zopen
from scipy.stats import norm
from pymatgen.core import Composition, Element, Molecule
from pymatgen.core.operations import SymmOp
from pymatgen.core.units import Ha_to_eV
from pymatgen.electronic_structure.core import Spin
from pymatgen.util.coord import get_angle
from pymatgen.util.plotting import pretty_plot
def to_input(self, mol=None, charge=None, spin_multiplicity=None, title=None, functional=None, basis_set=None, route_parameters=None, input_parameters=None, link0_parameters=None, dieze_tag=None, cart_coords=False):
    """
        Create a new input object using by default the last geometry read in
        the output file and with the same calculation parameters. Arguments
        are the same as GaussianInput class.

        Returns:
            gaunip (GaussianInput) : the gaussian input object
        """
    if not mol:
        mol = self.final_structure
    if charge is None:
        charge = self.charge
    if spin_multiplicity is None:
        spin_multiplicity = self.spin_multiplicity
    if not title:
        title = self.title
    if not functional:
        functional = self.functional
    if not basis_set:
        basis_set = self.basis_set
    if not route_parameters:
        route_parameters = self.route_parameters
    if not link0_parameters:
        link0_parameters = self.link0
    if not dieze_tag:
        dieze_tag = self.dieze_tag
    return GaussianInput(mol=mol, charge=charge, spin_multiplicity=spin_multiplicity, title=title, functional=functional, basis_set=basis_set, route_parameters=route_parameters, input_parameters=input_parameters, link0_parameters=link0_parameters, dieze_tag=dieze_tag)