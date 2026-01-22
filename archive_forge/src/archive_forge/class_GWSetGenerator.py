from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.symmetry.bandstructure import HighSymmKpath
@dataclass
class GWSetGenerator(AimsInputGenerator):
    """
    A generator for the input set for calculations employing GW self-energy correction.

    Parameters
    ----------
    calc_type: str
        The type of calculations
    k_point_density: float
        The number of k_points per angstrom
    """
    calc_type: str = 'GW'
    k_point_density: float = 20

    def get_parameter_updates(self, structure: Structure | Molecule, prev_parameters: dict[str, Any]) -> dict[str, Any]:
        """Get the parameter updates for the calculation.

        Parameters
        ----------
        structure: Structure or Molecule
            The structure to calculate the bands for
        prev_parameters: Dict[str, Any]
            The previous parameters

        Returns:
            dict: The updated for the parameters for the output section of FHI-aims
        """
        updates = {'anacon_type': 'two-pole'}
        current_output = prev_parameters.get('output', [])
        if isinstance(structure, Structure) and all(structure.lattice.pbc):
            updates.update(qpe_calc='gw_expt', output=current_output + prepare_band_input(structure, self.k_point_density))
        else:
            updates.update(qpe_calc='gw')
        return updates