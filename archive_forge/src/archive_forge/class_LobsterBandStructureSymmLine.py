from __future__ import annotations
import collections
import itertools
import math
import re
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff
class LobsterBandStructureSymmLine(BandStructureSymmLine):
    """Lobster subclass of BandStructure with customized functions."""

    def as_dict(self):
        """JSON-serializable dict representation of BandStructureSymmLine."""
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'lattice_rec': self.lattice_rec.as_dict(), 'efermi': self.efermi, 'kpoints': []}
        for k in self.kpoints:
            dct['kpoints'].append(k.as_dict()['fcoords'])
        dct['branches'] = self.branches
        dct['bands'] = {str(int(spin)): self.bands[spin].tolist() for spin in self.bands}
        dct['is_metal'] = self.is_metal()
        vbm = self.get_vbm()
        dct['vbm'] = {'energy': vbm['energy'], 'kpoint_index': [int(x) for x in vbm['kpoint_index']], 'band_index': {str(int(spin)): vbm['band_index'][spin] for spin in vbm['band_index']}, 'projections': {str(spin): v for spin, v in vbm['projections'].items()}}
        cbm = self.get_cbm()
        dct['cbm'] = {'energy': cbm['energy'], 'kpoint_index': [int(x) for x in cbm['kpoint_index']], 'band_index': {str(int(spin)): cbm['band_index'][spin] for spin in cbm['band_index']}, 'projections': {str(spin): v for spin, v in cbm['projections'].items()}}
        dct['band_gap'] = self.get_band_gap()
        dct['labels_dict'] = {}
        dct['is_spin_polarized'] = self.is_spin_polarized
        for c, label in self.labels_dict.items():
            mongo_key = c if not c.startswith('$') else ' ' + c
            dct['labels_dict'][mongo_key] = label.as_dict()['fcoords']
        if len(self.projections) != 0:
            dct['structure'] = self.structure.as_dict()
            dct['projections'] = {str(int(spin)): np.array(v).tolist() for spin, v in self.projections.items()}
        return dct

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct (dict): A dict with all data for a band structure symmetry line
                object.

        Returns:
            A BandStructureSymmLine object
        """
        try:
            labels_dict = {k.strip(): v for k, v in dct['labels_dict'].items()}
            projections = {}
            structure = None
            if dct.get('projections'):
                if isinstance(dct['projections']['1'][0][0], dict):
                    raise ValueError('Old band structure dict format detected!')
                structure = Structure.from_dict(dct['structure'])
                projections = {Spin(int(spin)): np.array(v) for spin, v in dct['projections'].items()}
            return cls(dct['kpoints'], {Spin(int(k)): dct['bands'][k] for k in dct['bands']}, Lattice(dct['lattice_rec']['matrix']), dct['efermi'], labels_dict, structure=structure, projections=projections)
        except Exception:
            warnings.warn('Trying from_dict failed. Now we are trying the old format. Please convert your BS dicts to the new format. The old format will be retired in pymatgen 5.0.')
            return cls.from_old_dict(dct)

    @classmethod
    def from_old_dict(cls, dct) -> Self:
        """
        Args:
            dct (dict): A dict with all data for a band structure symmetry line
                object.

        Returns:
            A BandStructureSymmLine object
        """
        labels_dict = {k.strip(): v for k, v in dct['labels_dict'].items()}
        projections: dict = {}
        structure = None
        if 'projections' in dct and len(dct['projections']) != 0:
            structure = Structure.from_dict(dct['structure'])
            projections = {}
            for spin in dct['projections']:
                dd = []
                for i in range(len(dct['projections'][spin])):
                    ddd = []
                    for j in range(len(dct['projections'][spin][i])):
                        ddd.append(dct['projections'][spin][i][j])
                    dd.append(np.array(ddd))
                projections[Spin(int(spin))] = np.array(dd)
        return cls(dct['kpoints'], {Spin(int(k)): dct['bands'][k] for k in dct['bands']}, Lattice(dct['lattice_rec']['matrix']), dct['efermi'], labels_dict, structure=structure, projections=projections)

    def get_projection_on_elements(self):
        """Method returning a dictionary of projections on elements.
        It sums over all available orbitals for each element.

        Returns:
            a dictionary in the {Spin.up:[][{Element:values}],
            Spin.down:[][{Element:values}]} format
            if there is no projections in the band structure
            returns an empty dict
        """
        result = {}
        for spin, v in self.projections.items():
            result[spin] = [[collections.defaultdict(float) for i in range(len(self.kpoints))] for j in range(self.nb_bands)]
            for i, j in itertools.product(range(self.nb_bands), range(len(self.kpoints))):
                for key, item in v[i][j].items():
                    for item2 in item.values():
                        specie = str(Element(re.split('[0-9]+', key)[0]))
                        result[spin][i][j][specie] += item2
        return result

    def get_projections_on_elements_and_orbitals(self, el_orb_spec):
        """Method returning a dictionary of projections on elements and specific
        orbitals.

        Args:
            el_orb_spec: A dictionary of Elements and Orbitals for which we want
                to have projections on. It is given as: {Element:[orbitals]},
                e.g., {'Si':['3s','3p']} or {'Si':['3s','3p_x', '3p_y', '3p_z']} depending on input files

        Returns:
            A dictionary of projections on elements in the
            {Spin.up:[][{Element:{orb:values}}],
            Spin.down:[][{Element:{orb:values}}]} format
            if there is no projections in the band structure returns an empty
            dict.
        """
        result = {}
        el_orb_spec = {get_el_sp(el): orbs for el, orbs in el_orb_spec.items()}
        for spin, v in self.projections.items():
            result[spin] = [[{str(e): collections.defaultdict(float) for e in el_orb_spec} for i in range(len(self.kpoints))] for j in range(self.nb_bands)]
            for i, j in itertools.product(range(self.nb_bands), range(len(self.kpoints))):
                for key, item in v[i][j].items():
                    for key2, item2 in item.items():
                        specie = str(Element(re.split('[0-9]+', key)[0]))
                        if get_el_sp(str(specie)) in el_orb_spec and key2 in el_orb_spec[get_el_sp(str(specie))]:
                            result[spin][i][j][specie][key2] += item2
        return result