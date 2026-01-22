from __future__ import annotations
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from monty.json import MontyDecoder, MSONable, jsanitize
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.voronoi import DetailedVoronoiContainer
from pymatgen.analysis.chemenv.utils.chemenv_errors import ChemenvError
from pymatgen.analysis.chemenv.utils.defs_utils import AdditionalConditions
from pymatgen.core import Element, PeriodicNeighbor, PeriodicSite, Species, Structure
class ChemicalEnvironments(MSONable):
    """
    Class used to store all the information about the chemical environment of a given site for a given list of
    coordinated neighbors (internally called "cn_map").
    """

    def __init__(self, coord_geoms=None):
        """
        Initializes the ChemicalEnvironments object containing all the information about the chemical
        environment of a given site.

        Args:
            coord_geoms: coordination geometries to be added to the chemical environment.
        """
        if coord_geoms is None:
            self.coord_geoms = {}
        else:
            raise NotImplementedError('Constructor for ChemicalEnvironments with the coord_geoms argument is not yet implemented')

    def __getitem__(self, mp_symbol):
        return self.coord_geoms[mp_symbol]

    def __len__(self):
        """
        Returns the number of coordination geometries in this ChemicalEnvironments object.

        Returns:
            Number of coordination geometries in this ChemicalEnvironments object.
        """
        return len(self.coord_geoms)

    def __iter__(self):
        yield from self.coord_geoms.items()

    def minimum_geometry(self, symmetry_measure_type=None, max_csm=None):
        """
        Returns the geometry with the minimum continuous symmetry measure of this ChemicalEnvironments.

        Returns:
            tuple (symbol, csm) with symbol being the geometry with the minimum continuous symmetry measure and
            csm being the continuous symmetry measure associated to it.

        Raises:
            ValueError if no coordination geometry is found in this ChemicalEnvironments object.
        """
        if len(self.coord_geoms) == 0:
            return None
        cglist = list(self.coord_geoms)
        if symmetry_measure_type is None:
            csms = np.array([self.coord_geoms[cg]['other_symmetry_measures']['csm_wcs_ctwcc'] for cg in cglist])
        else:
            csms = np.array([self.coord_geoms[cg]['other_symmetry_measures'][symmetry_measure_type] for cg in cglist])
        csmlist = [self.coord_geoms[cg] for cg in cglist]
        imin = np.argmin(csms)
        if max_csm is not None and csmlist[imin] > max_csm:
            return None
        return (cglist[imin], csmlist[imin])

    def minimum_geometries(self, n=None, symmetry_measure_type=None, max_csm=None):
        """
        Returns a list of geometries with increasing continuous symmetry measure in this ChemicalEnvironments object.

        Args:
            n: Number of geometries to be included in the list.

        Returns:
            List of geometries with increasing continuous symmetry measure in this ChemicalEnvironments object.

        Raises:
            ValueError if no coordination geometry is found in this ChemicalEnvironments object.
        """
        cglist = list(self.coord_geoms)
        if symmetry_measure_type is None:
            csms = np.array([self.coord_geoms[cg]['other_symmetry_measures']['csm_wcs_ctwcc'] for cg in cglist])
        else:
            csms = np.array([self.coord_geoms[cg]['other_symmetry_measures'][symmetry_measure_type] for cg in cglist])
        csmlist = [self.coord_geoms[cg] for cg in cglist]
        isorted = np.argsort(csms)
        if max_csm is not None:
            if n is None:
                return [(cglist[ii], csmlist[ii]) for ii in isorted if csms[ii] <= max_csm]
            return [(cglist[ii], csmlist[ii]) for ii in isorted[:n] if csms[ii] <= max_csm]
        if n is None:
            return [(cglist[ii], csmlist[ii]) for ii in isorted]
        return [(cglist[ii], csmlist[ii]) for ii in isorted[:n]]

    def add_coord_geom(self, mp_symbol, symmetry_measure, algo='UNKNOWN', permutation=None, override=False, local2perfect_map=None, perfect2local_map=None, detailed_voronoi_index=None, other_symmetry_measures=None, rotation_matrix=None, scaling_factor=None):
        """
        Adds a coordination geometry to the ChemicalEnvironments object.

        Args:
            mp_symbol: Symbol of the coordination geometry added.
            symmetry_measure: Symmetry measure of the coordination geometry added.
            algo: Algorithm used for the search of the coordination geometry added.
            permutation: Permutation of the neighbors that leads to the csm stored.
            override: If set to True, the coordination geometry will override the existent one if present.
            local2perfect_map: Mapping of the local indices to the perfect indices.
            perfect2local_map: Mapping of the perfect indices to the local indices.
            detailed_voronoi_index: Index in the voronoi containing the neighbors set.
            other_symmetry_measures: Other symmetry measure of the coordination geometry added (with/without the
                central atom, centered on the central atom or on the centroid with/without the central atom).
            rotation_matrix: Rotation matrix mapping the local geometry to the perfect geometry.
            scaling_factor: Scaling factor mapping the local geometry to the perfect geometry.

        Raises:
            ChemenvError if the coordination geometry is already added and override is set to False
        """
        if not all_cg.is_a_valid_coordination_geometry(mp_symbol=mp_symbol):
            raise ChemenvError(self.__class__, 'add_coord_geom', f'Coordination geometry with mp_symbol {mp_symbol!r} is not valid')
        if mp_symbol in list(self.coord_geoms) and (not override):
            raise ChemenvError(self.__class__, 'add_coord_geom', 'This coordination geometry is already present and override is set to False')
        self.coord_geoms[mp_symbol] = {'symmetry_measure': float(symmetry_measure), 'algo': algo, 'permutation': [int(i) for i in permutation], 'local2perfect_map': local2perfect_map, 'perfect2local_map': perfect2local_map, 'detailed_voronoi_index': detailed_voronoi_index, 'other_symmetry_measures': other_symmetry_measures, 'rotation_matrix': rotation_matrix, 'scaling_factor': scaling_factor}

    def __str__(self):
        """
        Returns a string representation of the ChemicalEnvironments object.

        Returns:
            String representation of the ChemicalEnvironments object.
        """
        out = 'Chemical environments object :\n'
        if len(self.coord_geoms) == 0:
            out += ' => No coordination in it <=\n'
            return out
        for key in self.coord_geoms:
            mp_symbol = key
            break
        cn = symbol_cn_mapping[mp_symbol]
        out += f' => Coordination {cn} <=\n'
        mp_symbols = list(self.coord_geoms)
        csms_wcs = [self.coord_geoms[mp_symbol]['other_symmetry_measures']['csm_wcs_ctwcc'] for mp_symbol in mp_symbols]
        icsms_sorted = np.argsort(csms_wcs)
        mp_symbols = [mp_symbols[ii] for ii in icsms_sorted]
        for mp_symbol in mp_symbols:
            csm_wcs = self.coord_geoms[mp_symbol]['other_symmetry_measures']['csm_wcs_ctwcc']
            csm_wocs = self.coord_geoms[mp_symbol]['other_symmetry_measures']['csm_wocs_ctwocc']
            out += f'   - {mp_symbol}\n'
            out += f'      csm1 (with central site) : {csm_wcs}'
            out += f'      csm2 (without central site) : {csm_wocs}'
            out += f'     algo : {self.coord_geoms[mp_symbol]['algo']}'
            out += f'     perm : {self.coord_geoms[mp_symbol]['permutation']}\n'
            out += f'       local2perfect : {self.coord_geoms[mp_symbol]['local2perfect_map']}\n'
            out += f'       perfect2local : {self.coord_geoms[mp_symbol]['perfect2local_map']}\n'
        return out

    def is_close_to(self, other, rtol=0.0, atol=1e-08) -> bool:
        """
        Whether this ChemicalEnvironments object is close to another one.

        Args:
            other: Another ChemicalEnvironments object.
            rtol: Relative tolerance for the comparison of Continuous Symmetry Measures.
            atol: Absolute tolerance for the comparison of Continuous Symmetry Measures.

        Returns:
            bool: True if the two ChemicalEnvironments objects are close to each other.
        """
        if set(self.coord_geoms) != set(other.coord_geoms):
            return False
        for mp_symbol, cg_dict_self in self.coord_geoms.items():
            cg_dict_other = other[mp_symbol]
            other_csms_self = cg_dict_self['other_symmetry_measures']
            other_csms_other = cg_dict_other['other_symmetry_measures']
            for csmtype in ['csm_wcs_ctwcc', 'csm_wcs_ctwocc', 'csm_wcs_csc', 'csm_wocs_ctwcc', 'csm_wocs_ctwocc', 'csm_wocs_csc']:
                if not np.isclose(other_csms_self[csmtype], other_csms_other[csmtype], rtol=rtol, atol=atol):
                    return False
        return True

    def __eq__(self, other: object) -> bool:
        """
        Equality method that checks if the ChemicalEnvironments object is equal to another ChemicalEnvironments.
        object.

        Args:
            other: ChemicalEnvironments object to compare with.

        Returns:
            bool: True if both objects are equal.
        """
        if not isinstance(other, ChemicalEnvironments):
            return NotImplemented
        if set(self.coord_geoms) != set(other.coord_geoms):
            return False
        for mp_symbol, cg_dict_self in self.coord_geoms.items():
            cg_dict_other = other.coord_geoms[mp_symbol]
            if cg_dict_self['symmetry_measure'] != cg_dict_other['symmetry_measure']:
                return False
            if cg_dict_self['algo'] != cg_dict_other['algo']:
                return False
            if cg_dict_self['permutation'] != cg_dict_other['permutation']:
                return False
            if cg_dict_self['detailed_voronoi_index'] != cg_dict_other['detailed_voronoi_index']:
                return False
            other_csms_self = cg_dict_self['other_symmetry_measures']
            other_csms_other = cg_dict_other['other_symmetry_measures']
            for csmtype in ['csm_wcs_ctwcc', 'csm_wcs_ctwocc', 'csm_wcs_csc', 'csm_wocs_ctwcc', 'csm_wocs_ctwocc', 'csm_wocs_csc']:
                if other_csms_self[csmtype] != other_csms_other[csmtype]:
                    return False
        return True

    def as_dict(self):
        """
        Returns a dictionary representation of the ChemicalEnvironments object.

        Returns:
            A dictionary representation of the ChemicalEnvironments object.
        """
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'coord_geoms': jsanitize(self.coord_geoms)}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Reconstructs the ChemicalEnvironments object from a dict representation of the ChemicalEnvironments created
        using the as_dict method.

        Args:
            dct: dict representation of the ChemicalEnvironments object.

        Returns:
            ChemicalEnvironments object.
        """
        ce = cls()
        for cg in dct['coord_geoms']:
            if dct['coord_geoms'][cg]['local2perfect_map'] is None:
                l2p_map = None
            else:
                l2p_map = {int(key): int(val) for key, val in dct['coord_geoms'][cg]['local2perfect_map'].items()}
            if dct['coord_geoms'][cg]['perfect2local_map'] is None:
                p2l_map = None
            else:
                p2l_map = {int(key): int(val) for key, val in dct['coord_geoms'][cg]['perfect2local_map'].items()}
            if 'other_symmetry_measures' in dct['coord_geoms'][cg] and dct['coord_geoms'][cg]['other_symmetry_measures'] is not None:
                other_csms = dct['coord_geoms'][cg]['other_symmetry_measures']
            else:
                other_csms = None
            ce.add_coord_geom(cg, dct['coord_geoms'][cg]['symmetry_measure'], dct['coord_geoms'][cg]['algo'], permutation=dct['coord_geoms'][cg]['permutation'], local2perfect_map=l2p_map, perfect2local_map=p2l_map, detailed_voronoi_index=dct['coord_geoms'][cg]['detailed_voronoi_index'], other_symmetry_measures=other_csms, rotation_matrix=dct['coord_geoms'][cg]['rotation_matrix'], scaling_factor=dct['coord_geoms'][cg]['scaling_factor'])
        return ce