from __future__ import annotations
import logging
import os
import subprocess
import warnings
from enum import Enum, unique
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from monty.serialization import loadfn
from monty.tempfile import ScratchDir
from scipy.spatial import KDTree
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import DummySpecies
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar, VolumetricData
from pymatgen.util.due import Doi, due
class Critic2Analysis(MSONable):
    """Class to process the standard output from critic2 into pymatgen-compatible objects."""

    def __init__(self, structure: Structure, stdout: str | None=None, stderr: str | None=None, cpreport: dict | None=None, yt: dict | None=None, zpsp: dict | None=None) -> None:
        """This class is used to store results from the Critic2Caller.

        To explore the bond graph, use the "structure_graph"
        method, which returns a user-friendly StructureGraph
        class with bonding information. By default, this returns
        a StructureGraph with edge weights as bond lengths, but
        can optionally return a graph with edge weights as any
        property supported by the `CriticalPoint` class, such as
        bond ellipticity.

        This class also provides an interface to explore just the
        non-symmetrically-equivalent critical points via the
        `critical_points` attribute, and also all critical
        points (via nodes dict) and connections between them
        (via edges dict). The user should be familiar with critic2
        before trying to understand these.

        Indexes of nucleus critical points in the nodes dict are the
        same as the corresponding sites in structure, with indices of
        other critical points arbitrarily assigned.

        Only one of (stdout, cpreport) required, with cpreport preferred
        since this is a new, native JSON output from critic2.

        Args:
            structure: associated Structure
            stdout: stdout from running critic2 in automatic mode
            stderr: stderr from running critic2 in automatic mode
            cpreport: json output from CPREPORT command
            yt: json output from YT command
            zpsp (dict): Dict of element/symbol name to number of electrons
                (ZVAL in VASP pseudopotential), with which to calculate charge transfer.
                Optional.

        Args:
            structure (Structure): Associated Structure.
            stdout (str, optional): stdout from running critic2 in automatic mode.
            stderr (str, optional): stderr from running critic2 in automatic mode.
            cpreport (dict, optional): JSON output from CPREPORT command. Either this or stdout required.
            yt (dict, optional): JSON output from YT command.
            zpsp (dict, optional): Dict of element/symbol name to number of electrons (ZVAL in VASP pseudopotential),
                with which to calculate charge transfer. Optional.

        Raises:
            ValueError: If one of cpreport or stdout is not provided.
        """
        self.structure = structure
        self._stdout = stdout
        self._stderr = stderr
        self._cpreport = cpreport
        self._yt = yt
        self._zpsp = zpsp
        self.nodes: dict[int, dict] = {}
        self.edges: dict[int, dict] = {}
        if yt:
            self.structure = self._annotate_structure_with_yt(yt, structure, zpsp)
        if cpreport:
            self._parse_cpreport(cpreport)
        elif stdout:
            self._parse_stdout(stdout)
        else:
            raise ValueError('One of cpreport or stdout required.')
        self._remap_indices()

    def structure_graph(self, include_critical_points=('bond', 'ring', 'cage')):
        """A StructureGraph object describing bonding information in the crystal.

        Args:
            include_critical_points: add DummySpecies for the critical points themselves, a list of
                "nucleus", "bond", "ring", "cage", set to None to disable

        Returns:
            StructureGraph
        """
        structure = self.structure.copy()
        point_idx_to_struct_idx = {}
        if include_critical_points:
            for prop in ('ellipticity', 'laplacian', 'field'):
                structure.add_site_property(prop, [0] * len(structure))
            for idx, node in self.nodes.items():
                cp = self.critical_points[node['unique_idx']]
                if cp.type.value in include_critical_points:
                    specie = DummySpecies(f'X{cp.type.value[0]}cp', oxidation_state=None)
                    structure.append(specie, node['frac_coords'], properties={'ellipticity': cp.ellipticity, 'laplacian': cp.laplacian, 'field': cp.field})
                    point_idx_to_struct_idx[idx] = len(structure) - 1
        edge_weight = 'bond_length'
        edge_weight_units = 'Å'
        struct_graph = StructureGraph.from_empty_graph(structure, name='bonds', edge_weight_name=edge_weight, edge_weight_units=edge_weight_units)
        edges = self.edges.copy()
        idx_to_delete = []
        for idx, edge in edges.items():
            unique_idx = self.nodes[idx]['unique_idx']
            if self.critical_points[unique_idx].type == CriticalPointType.bond and idx not in idx_to_delete:
                for idx2, edge2 in edges.items():
                    if idx != idx2 and edge == edge2:
                        idx_to_delete.append(idx2)
                        warnings.warn('Duplicate edge detected, try re-running critic2 with custom parameters to fix this. Mostly harmless unless user is also interested in rings/cages.')
                        logger.debug(f'Duplicate edge between points {idx} (unique point {self.nodes[idx]['unique_idx']})and {idx2} ({self.nodes[idx2]['unique_idx']}).')
        for idx in idx_to_delete:
            del edges[idx]
        for idx, edge in edges.items():
            unique_idx = self.nodes[idx]['unique_idx']
            if self.critical_points[unique_idx].type == CriticalPointType.bond:
                from_idx = edge['from_idx']
                to_idx = edge['to_idx']
                skip_bond = False
                if include_critical_points and 'nnattr' not in include_critical_points:
                    from_type = self.critical_points[self.nodes[from_idx]['unique_idx']].type
                    to_type = self.critical_points[self.nodes[from_idx]['unique_idx']].type
                    skip_bond = from_type != CriticalPointType.nucleus or to_type != CriticalPointType.nucleus
                if not skip_bond:
                    from_lvec = edge['from_lvec']
                    to_lvec = edge['to_lvec']
                    relative_lvec = np.subtract(to_lvec, from_lvec)
                    struct_from_idx = point_idx_to_struct_idx.get(from_idx, from_idx)
                    struct_to_idx = point_idx_to_struct_idx.get(to_idx, to_idx)
                    weight = self.structure.get_distance(struct_from_idx, struct_to_idx, jimage=relative_lvec)
                    crit_point = self.critical_points[unique_idx]
                    edge_properties = {'field': crit_point.field, 'laplacian': crit_point.laplacian, 'ellipticity': crit_point.ellipticity, 'frac_coords': self.nodes[idx]['frac_coords']}
                    struct_graph.add_edge(struct_from_idx, struct_to_idx, from_jimage=from_lvec, to_jimage=to_lvec, weight=weight, edge_properties=edge_properties)
        return struct_graph

    def get_critical_point_for_site(self, n: int):
        """
        Args:
            n (int): Site index.

        Returns:
            CriticalPoint
        """
        return self.critical_points[self.nodes[n]['unique_idx']]

    def get_volume_and_charge_for_site(self, idx):
        """
        Args:
            idx: Site index.

        Returns:
            dict: with "volume" and "charge" keys, or None if YT integration not performed
        """
        if not self._node_values:
            return None
        return self._node_values[idx]

    def _parse_cpreport(self, cpreport):

        def get_type(signature: int, is_nucleus: bool):
            if signature == 3:
                return 'cage'
            if signature == 1:
                return 'ring'
            if signature == -1:
                return 'bond'
            if signature == -3:
                if is_nucleus:
                    return 'nucleus'
                return 'nnattr'
            return None
        bohr_to_angstrom = 0.529177
        self.critical_points = [CriticalPoint(p['id'] - 1, get_type(p['signature'], p['is_nucleus']), p['fractional_coordinates'], p['point_group'], p['multiplicity'], p['field'], p['gradient'], coords=[x * bohr_to_angstrom for x in p['cartesian_coordinates']] if cpreport['units'] == 'bohr' else None, field_hessian=p['hessian']) for p in cpreport['critical_points']['nonequivalent_cps']]
        for point in cpreport['critical_points']['cell_cps']:
            self._add_node(idx=point['id'] - 1, unique_idx=point['nonequivalent_id'] - 1, frac_coords=point['fractional_coordinates'])
            if 'attractors' in point:
                self._add_edge(idx=point['id'] - 1, from_idx=int(point['attractors'][0]['cell_id']) - 1, from_lvec=point['attractors'][0]['lvec'], to_idx=int(point['attractors'][1]['cell_id']) - 1, to_lvec=point['attractors'][1]['lvec'])

    def _remap_indices(self):
        """Re-maps indices on self.nodes and self.edges such that node indices match
        that of structure, and then sorts self.nodes by index.
        """
        node_mapping = {}
        frac_coords = np.array(self.structure.frac_coords) % 1
        kd = KDTree(frac_coords)
        node_mapping = {}
        for idx, node in self.nodes.items():
            if self.critical_points[node['unique_idx']].type == CriticalPointType.nucleus:
                node_mapping[idx] = kd.query(node['frac_coords'])[1]
        if len(node_mapping) != len(self.structure):
            warnings.warn(f'Check that all sites in input structure ({len(self.structure)}) have been detected by critic2 ({len(node_mapping)}).')
        self.nodes = {node_mapping.get(idx, idx): node for idx, node in self.nodes.items()}
        for edge in self.edges.values():
            edge['from_idx'] = node_mapping.get(edge['from_idx'], edge['from_idx'])
            edge['to_idx'] = node_mapping.get(edge['to_idx'], edge['to_idx'])

    @staticmethod
    def _annotate_structure_with_yt(yt, structure: Structure, zpsp):
        volume_idx = charge_idx = None
        for prop in yt['integration']['properties']:
            if prop['label'] == 'Volume':
                volume_idx = prop['id'] - 1
            elif prop['label'] == '$chg_int':
                charge_idx = prop['id'] - 1

        def get_volume_and_charge(nonequiv_idx):
            attractor = yt['integration']['attractors'][nonequiv_idx - 1]
            if attractor['id'] != nonequiv_idx:
                raise ValueError(f'List of attractors may be un-ordered (wanted id={nonequiv_idx}): {attractor}')
            return (attractor['integrals'][volume_idx], attractor['integrals'][charge_idx])
        volumes = []
        charges = []
        charge_transfer = []
        for idx, site in enumerate(yt['structure']['cell_atoms']):
            if not np.allclose(structure[idx].frac_coords, site['fractional_coordinates']):
                raise IndexError(f"Site in structure doesn't seem to match site in YT integration:\n{structure[idx]}\n{site}")
            volume, charge = get_volume_and_charge(site['nonequivalent_id'])
            volumes.append(volume)
            charges.append(charge)
            if zpsp:
                if structure[idx].species_string in zpsp:
                    charge_transfer.append(charge - zpsp[structure[idx].species_string])
                else:
                    raise ValueError(f'ZPSP argument does not seem compatible with species in structure ({structure[idx].species_string}): {zpsp}')
        structure = structure.copy()
        structure.add_site_property('bader_volume', volumes)
        structure.add_site_property('bader_charge', charges)
        if zpsp:
            if len(charge_transfer) != len(charges):
                warnings.warn(f'Something went wrong calculating charge transfer: {charge_transfer}')
            else:
                structure.add_site_property('bader_charge_transfer', charge_transfer)
        return structure

    def _parse_stdout(self, stdout):
        warnings.warn('Parsing critic2 standard output is deprecated and will not be maintained, please use the native JSON output in future.')
        stdout = stdout.split('\n')
        unique_critical_points = []
        for idx, line in enumerate(stdout):
            if 'mult  name            f             |grad|           lap' in line:
                start_i = idx + 1
            elif '* Analysis of system bonds' in line:
                end_i = idx - 2
        for idx, line in enumerate(stdout):
            if start_i <= idx <= end_i:
                split = line.replace('(', '').replace(')', '').split()
                unique_idx = int(split[0]) - 1
                point_group = split[1]
                critical_point_type = split[3]
                frac_coords = [float(split[4]), float(split[5]), float(split[6])]
                multiplicity = float(split[7])
                field = float(split[9])
                field_gradient = float(split[10])
                point = CriticalPoint(unique_idx, critical_point_type, frac_coords, point_group, multiplicity, field, field_gradient)
                unique_critical_points.append(point)
        for idx, line in enumerate(stdout):
            if '+ Critical point no.' in line:
                unique_idx = int(line.split()[4]) - 1
            elif 'Hessian:' in line:
                l1 = list(map(float, stdout[idx + 1].split()))
                l2 = list(map(float, stdout[idx + 2].split()))
                l3 = list(map(float, stdout[idx + 3].split()))
                hessian = [[l1[0], l1[1], l1[2]], [l2[0], l2[1], l2[2]], [l3[0], l3[1], l3[2]]]
                unique_critical_points[unique_idx].field_hessian = hessian
        self.critical_points = unique_critical_points
        for idx, line in enumerate(stdout):
            if '#cp  ncp   typ        position ' in line:
                start_i = idx + 1
            elif '* Attractor connectivity matrix' in line:
                end_i = idx - 2
        for idx, line in enumerate(stdout):
            if start_i <= idx <= end_i:
                split = line.replace('(', '').replace(')', '').split()
                idx = int(split[0]) - 1
                unique_idx = int(split[1]) - 1
                frac_coords = [float(split[3]), float(split[4]), float(split[5])]
                self._add_node(idx, unique_idx, frac_coords)
                if len(split) > 6:
                    from_idx = int(split[6]) - 1
                    to_idx = int(split[10]) - 1
                    self._add_edge(idx, from_idx=from_idx, from_lvec=(int(split[7]), int(split[8]), int(split[9])), to_idx=to_idx, to_lvec=(int(split[11]), int(split[12]), int(split[13])))

    def _add_node(self, idx, unique_idx, frac_coords):
        """Add information about a node describing a critical point.

        Args:
            idx: index
            unique_idx: index of unique CriticalPoint,
                used to look up more information of point (field etc.)
            frac_coords: fractional coordinates of point
        """
        self.nodes[idx] = {'unique_idx': unique_idx, 'frac_coords': frac_coords}

    def _add_edge(self, idx, from_idx, from_lvec, to_idx, to_lvec):
        """Add information about an edge linking two critical points.

        This actually describes two edges:

        from_idx ------ idx ------ to_idx

        However, in practice, from_idx and to_idx will typically be
        atom nuclei, with the center node (idx) referring to a bond
        critical point. Thus, it will be more convenient to model
        this as a single edge linking nuclei with the properties
        of the bond critical point stored as an edge attribute.

        Args:
            idx: index of node
            from_idx: from index of node
            from_lvec: vector of lattice image the from node is in
                as tuple of ints
            to_idx: to index of node
            to_lvec: vector of lattice image the to node is in as
                tuple of ints
        """
        self.edges[idx] = {'from_idx': from_idx, 'from_lvec': from_lvec, 'to_idx': to_idx, 'to_lvec': to_lvec}