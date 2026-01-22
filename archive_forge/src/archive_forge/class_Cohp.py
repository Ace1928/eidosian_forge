from __future__ import annotations
import re
import sys
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.io.lmto import LMTOCopl
from pymatgen.io.lobster import Cohpcar
from pymatgen.util.coord import get_linear_interpolated_value
from pymatgen.util.due import Doi, due
from pymatgen.util.num import round_to_sigfigs
class Cohp(MSONable):
    """Basic COHP object."""

    def __init__(self, efermi, energies, cohp, are_coops=False, are_cobis=False, are_multi_center_cobis=False, icohp=None) -> None:
        """
        Args:
            are_coops: Indicates whether this object describes COOPs.
            are_cobis: Indicates whether this object describes COBIs.
            are_multi_center_cobis: Indicates whether this object describes multi-center COBIs
            efermi: Fermi energy.
            energies: A sequence of energies.
            cohp ({Spin: np.array}): representing the COHP for each spin.
            icohp ({Spin: np.array}): representing the ICOHP for each spin.
        """
        self.are_coops = are_coops
        self.are_cobis = are_cobis
        self.are_multi_center_cobis = are_multi_center_cobis
        self.efermi = efermi
        self.energies = np.array(energies)
        self.cohp = cohp
        self.icohp = icohp

    def __repr__(self) -> str:
        """Returns a string that can be easily plotted (e.g. using gnuplot)."""
        if self.are_coops:
            cohp_str = 'COOP'
        elif self.are_cobis or self.are_multi_center_cobis:
            cohp_str = 'COBI'
        else:
            cohp_str = 'COHP'
        header = ['Energy', f'{cohp_str}Up']
        data = [self.energies, self.cohp[Spin.up]]
        if Spin.down in self.cohp:
            header.append(f'{cohp_str}Down')
            data.append(self.cohp[Spin.down])
        if self.icohp:
            header.append(f'I{cohp_str}Up')
            data.append(self.icohp[Spin.up])
            if Spin.down in self.cohp:
                header.append(f'I{cohp_str}Down')
                data.append(self.icohp[Spin.down])
        format_header = '#' + ' '.join(('{:15s}' for __ in header))
        format_data = ' '.join(('{:.5f}' for __ in header))
        str_arr = [format_header.format(*header)]
        for idx in range(len(self.energies)):
            str_arr.append(format_data.format(*(d[idx] for d in data)))
        return '\n'.join(str_arr)

    def as_dict(self):
        """JSON-serializable dict representation of COHP."""
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'are_coops': self.are_coops, 'are_cobis': self.are_cobis, 'are_multi_center_cobis': self.are_multi_center_cobis, 'efermi': self.efermi, 'energies': self.energies.tolist(), 'COHP': {str(spin): pops.tolist() for spin, pops in self.cohp.items()}}
        if self.icohp:
            dct['ICOHP'] = {str(spin): pops.tolist() for spin, pops in self.icohp.items()}
        return dct

    def get_cohp(self, spin=None, integrated=False):
        """Returns the COHP or ICOHP for a particular spin.

        Args:
            spin: Spin. Can be parsed as spin object, integer (-1/1)
                or str ("up"/"down")
            integrated: Return COHP (False) or ICOHP (True)

        Returns:
            Returns the CHOP or ICOHP for the input spin. If Spin is
            None and both spins are present, both spins will be returned
            as a dictionary.
        """
        populations = self.cohp if not integrated else self.icohp
        if populations is None:
            return None
        if spin is None:
            return populations
        if isinstance(spin, int):
            spin = Spin(spin)
        elif isinstance(spin, str):
            spin = Spin({'up': 1, 'down': -1}[spin.lower()])
        return {spin: populations[spin]}

    def get_icohp(self, spin=None):
        """Convenient alternative to get the ICOHP for a particular spin."""
        return self.get_cohp(spin=spin, integrated=True)

    def get_interpolated_value(self, energy, integrated=False):
        """Returns the COHP for a particular energy.

        Args:
            energy: Energy to return the COHP value for.
            integrated: Return COHP (False) or ICOHP (True)
        """
        inter = {}
        for spin in self.cohp:
            if not integrated:
                inter[spin] = get_linear_interpolated_value(self.energies, self.cohp[spin], energy)
            elif self.icohp is not None:
                inter[spin] = get_linear_interpolated_value(self.energies, self.icohp[spin], energy)
            else:
                raise ValueError('ICOHP is empty.')
        return inter

    def has_antibnd_states_below_efermi(self, spin=None, limit=0.01):
        """Returns dict indicating if there are antibonding states below the Fermi level depending on the spin
        spin: Spin
        limit: -COHP smaller -limit will be considered.
        """
        populations = self.cohp
        n_energies_below_efermi = len([x for x in self.energies if x <= self.efermi])
        if populations is None:
            return None
        if spin is None:
            dict_to_return = {}
            for sp, cohp_vals in populations.items():
                if max(cohp_vals[0:n_energies_below_efermi]) > limit:
                    dict_to_return[sp] = True
                else:
                    dict_to_return[sp] = False
        else:
            dict_to_return = {}
            if isinstance(spin, int):
                spin = Spin(spin)
            elif isinstance(spin, str):
                spin = Spin({'up': 1, 'down': -1}[spin.lower()])
            if max(populations[spin][0:n_energies_below_efermi]) > limit:
                dict_to_return[spin] = True
            else:
                dict_to_return[spin] = False
        return dict_to_return

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """Returns a COHP object from a dict representation of the COHP."""
        icohp = {Spin(int(key)): np.array(val) for key, val in dct['ICOHP'].items()} if 'ICOHP' in dct else None
        are_cobis = dct.get('are_cobis', False)
        are_multi_center_cobis = dct.get('are_multi_center_cobis', False)
        return cls(dct['efermi'], dct['energies'], {Spin(int(key)): np.array(val) for key, val in dct['COHP'].items()}, icohp=icohp, are_coops=dct['are_coops'], are_cobis=are_cobis, are_multi_center_cobis=are_multi_center_cobis)