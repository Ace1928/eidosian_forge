from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.functools import lazy_property
from monty.json import MSONable
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from pymatgen.util.coord import get_linear_interpolated_value
class CompletePhononDos(PhononDos):
    """This wrapper class defines a total dos, and also provides a list of PDos.

    Attributes:
        pdos (dict): Dict of partial densities of the form {Site:Densities}.
            Densities are a dict of {Orbital:Values} where Values are a list of floats.
            Site is a pymatgen.core.sites.Site object.
    """

    def __init__(self, structure: Structure, total_dos, ph_doses: dict) -> None:
        """
        Args:
            structure: Structure associated with this particular DOS.
            total_dos: total Dos for structure
            ph_doses: The phonon DOSes are supplied as a dict of {Site: Densities}.
        """
        super().__init__(frequencies=total_dos.frequencies, densities=total_dos.densities)
        self.pdos = {site: np.array(dens) for site, dens in ph_doses.items()}
        self.structure = structure

    def get_site_dos(self, site) -> PhononDos:
        """Get the Dos for a site.

        Args:
            site: Site in Structure associated with CompletePhononDos.

        Returns:
            PhononDos: containing summed orbital densities for site.
        """
        return PhononDos(self.frequencies, self.pdos[site])

    def get_element_dos(self) -> dict:
        """Get element projected Dos.

        Returns:
            dict of {Element: Dos}
        """
        el_dos = {}
        for site, atom_dos in self.pdos.items():
            el = site.specie
            if el not in el_dos:
                el_dos[el] = np.array(atom_dos)
            else:
                el_dos[el] += np.array(atom_dos)
        return {el: PhononDos(self.frequencies, densities) for el, densities in el_dos.items()}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Returns CompleteDos object from dict representation."""
        total_dos = PhononDos.from_dict(dct)
        struct = Structure.from_dict(dct['structure'])
        ph_doses = dict(zip(struct, dct['pdos']))
        return cls(struct, total_dos, ph_doses)

    def as_dict(self):
        """JSON-serializable dict representation of CompletePhononDos."""
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'structure': self.structure.as_dict(), 'frequencies': list(self.frequencies), 'densities': list(self.densities), 'pdos': []}
        if len(self.pdos) > 0:
            for site in self.structure:
                dct['pdos'].append(list(self.pdos[site]))
        return dct

    def __str__(self) -> str:
        return f'Complete phonon DOS for {self.structure}'