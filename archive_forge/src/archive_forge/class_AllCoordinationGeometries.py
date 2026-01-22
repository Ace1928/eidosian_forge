from __future__ import annotations
import abc
import itertools
import json
import os
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MontyDecoder, MSONable
from scipy.special import factorial
class AllCoordinationGeometries(dict):
    """
    Class used to store all the reference "coordination geometries" (list with instances of the CoordinationGeometry
    classes).
    """

    def __init__(self, permutations_safe_override=False, only_symbols=None):
        """
        Initializes the list of Coordination Geometries.

        Args:
            permutations_safe_override: Whether to use safe permutations.
            only_symbols: Whether to restrict the list of environments to be identified.
        """
        dict.__init__(self)
        self.cg_list = []
        if only_symbols is None:
            with open(f'{module_dir}/coordination_geometries_files/allcg.txt') as file:
                data = file.readlines()
            for line in data:
                cg_file = f'{module_dir}/{line.strip()}'
                with open(cg_file) as file:
                    dd = json.load(file)
                self.cg_list.append(CoordinationGeometry.from_dict(dd))
        else:
            for symbol in only_symbols:
                fsymbol = symbol.replace(':', '#')
                cg_file = f'{module_dir}/coordination_geometries_files/{fsymbol}.json'
                with open(cg_file) as file:
                    dd = json.load(file)
                self.cg_list.append(CoordinationGeometry.from_dict(dd))
        self.cg_list.append(CoordinationGeometry(UNKNOWN_ENVIRONMENT_SYMBOL, 'Unknown environment', deactivate=True))
        self.cg_list.append(CoordinationGeometry(UNCLEAR_ENVIRONMENT_SYMBOL, 'Unclear environment', deactivate=True))
        if permutations_safe_override:
            for cg in self.cg_list:
                cg.permutations_safe_override = True
        self.minpoints = {}
        self.maxpoints = {}
        self.separations_cg = {}
        for cn in range(6, 21):
            for cg in self.get_implemented_geometries(coordination=cn):
                if only_symbols is not None and cg.ce_symbol not in only_symbols:
                    continue
                if cn not in self.separations_cg:
                    self.minpoints[cn] = 1000
                    self.maxpoints[cn] = 0
                    self.separations_cg[cn] = {}
                for algo in cg.algorithms:
                    sep = (len(algo.point_groups[0]), len(algo.plane_points), len(algo.point_groups[1]))
                    if sep not in self.separations_cg[cn]:
                        self.separations_cg[cn][sep] = []
                    self.separations_cg[cn][sep].append(cg.mp_symbol)
                    self.minpoints[cn] = min(self.minpoints[cn], algo.minimum_number_of_points)
                    self.maxpoints[cn] = max(self.maxpoints[cn], algo.maximum_number_of_points)
        self.maxpoints_inplane = {cn: max((sep[1] for sep in seps)) for cn, seps in self.separations_cg.items()}

    def __getitem__(self, key):
        return self.get_geometry_from_mp_symbol(key)

    def __contains__(self, item):
        try:
            self[item]
            return True
        except LookupError:
            return False

    def __repr__(self):
        """Returns a string with the list of coordination geometries."""
        outs = ['', '#=================================#', '# List of coordination geometries #', '#=================================#', '']
        for cg in self.cg_list:
            outs.append(repr(cg))
        return '\n'.join(outs)

    def __str__(self):
        """Returns a string with the list of coordination geometries that are implemented."""
        outs = ['', '#=======================================================#', '# List of coordination geometries currently implemented #', '#=======================================================#', '']
        for cg in self.cg_list:
            if cg.is_implemented():
                outs.append(str(cg))
        return '\n'.join(outs)

    def get_geometries(self, coordination=None, returned='cg'):
        """
        Returns a list of coordination geometries with the given coordination number.

        Args:
            coordination: The coordination number of which the list of coordination geometries are returned.
            returned: Type of objects in the list.
        """
        geom = []
        if coordination is None:
            for gg in self.cg_list:
                if returned == 'cg':
                    geom.append(gg)
                elif returned == 'mp_symbol':
                    geom.append(gg.mp_symbol)
        else:
            for gg in self.cg_list:
                if gg.get_coordination_number() == coordination:
                    if returned == 'cg':
                        geom.append(gg)
                    elif returned == 'mp_symbol':
                        geom.append(gg.mp_symbol)
        return geom

    def get_symbol_name_mapping(self, coordination=None):
        """
        Return a dictionary mapping the symbol of a CoordinationGeometry to its name.

        Args:
            coordination: Whether to restrict the dictionary to a given coordination.

        Returns:
            dict: map symbol of a CoordinationGeometry to its name.
        """
        geom = {}
        if coordination is None:
            for gg in self.cg_list:
                geom[gg.mp_symbol] = gg.name
        else:
            for gg in self.cg_list:
                if gg.get_coordination_number() == coordination:
                    geom[gg.mp_symbol] = gg.name
        return geom

    def get_symbol_cn_mapping(self, coordination=None):
        """
        Return a dictionary mapping the symbol of a CoordinationGeometry to its coordination.

        Args:
            coordination: Whether to restrict the dictionary to a given coordination.

        Returns:
            dict: map of symbol of a CoordinationGeometry to its coordination.
        """
        geom = {}
        if coordination is None:
            for gg in self.cg_list:
                geom[gg.mp_symbol] = gg.coordination_number
        else:
            for gg in self.cg_list:
                if gg.get_coordination_number() == coordination:
                    geom[gg.mp_symbol] = gg.coordination_number
        return geom

    def get_implemented_geometries(self, coordination=None, returned='cg', include_deactivated=False):
        """
        Returns a list of the implemented coordination geometries with the given coordination number.

        Args:
            coordination: The coordination number of which the list of implemented coordination geometries
                are returned.
            returned: Type of objects in the list.
            include_deactivated: Whether to include CoordinationGeometry that are deactivated.
        """
        geom = []
        if coordination is None:
            for gg in self.cg_list:
                if gg.points is not None and (not gg.deactivate or include_deactivated):
                    if returned == 'cg':
                        geom.append(gg)
                    elif returned == 'mp_symbol':
                        geom.append(gg.mp_symbol)
        else:
            for gg in self.cg_list:
                if gg.get_coordination_number() == coordination and gg.points is not None and (not gg.deactivate or include_deactivated):
                    if returned == 'cg':
                        geom.append(gg)
                    elif returned == 'mp_symbol':
                        geom.append(gg.mp_symbol)
        return geom

    def get_not_implemented_geometries(self, coordination=None, returned='mp_symbol'):
        """
        Returns a list of the implemented coordination geometries with the given coordination number.

        Args:
            coordination: The coordination number of which the list of implemented coordination geometries
                are returned.
            returned: Type of objects in the list.
        """
        geom = []
        if coordination is None:
            for gg in self.cg_list:
                if gg.points is None:
                    if returned == 'cg':
                        geom.append(gg)
                    elif returned == 'mp_symbol':
                        geom.append(gg.mp_symbol)
        else:
            for gg in self.cg_list:
                if gg.get_coordination_number() == coordination and gg.points is None:
                    if returned == 'cg':
                        geom.append(gg)
                    elif returned == 'mp_symbol':
                        geom.append(gg.mp_symbol)
        return geom

    def get_geometry_from_name(self, name):
        """
        Returns the coordination geometry of the given name.

        Args:
            name: The name of the coordination geometry.
        """
        for gg in self.cg_list:
            if gg.name == name or name in gg.alternative_names:
                return gg
        raise LookupError(f'No coordination geometry found with name {name!r}')

    def get_geometry_from_IUPAC_symbol(self, IUPAC_symbol):
        """
        Returns the coordination geometry of the given IUPAC symbol.

        Args:
            IUPAC_symbol: The IUPAC symbol of the coordination geometry.
        """
        for gg in self.cg_list:
            if gg.IUPAC_symbol == IUPAC_symbol:
                return gg
        raise LookupError(f'No coordination geometry found with IUPAC symbol {IUPAC_symbol!r}')

    def get_geometry_from_IUCr_symbol(self, IUCr_symbol):
        """
        Returns the coordination geometry of the given IUCr symbol.

        Args:
            IUCr_symbol: The IUCr symbol of the coordination geometry.
        """
        for gg in self.cg_list:
            if gg.IUCr_symbol == IUCr_symbol:
                return gg
        raise LookupError(f'No coordination geometry found with IUCr symbol {IUCr_symbol!r}')

    def get_geometry_from_mp_symbol(self, mp_symbol):
        """
        Returns the coordination geometry of the given mp_symbol.

        Args:
            mp_symbol: The mp_symbol of the coordination geometry.
        """
        for gg in self.cg_list:
            if gg.mp_symbol == mp_symbol:
                return gg
        raise LookupError(f'No coordination geometry found with mp_symbol {mp_symbol!r}')

    def is_a_valid_coordination_geometry(self, mp_symbol=None, IUPAC_symbol=None, IUCr_symbol=None, name=None, cn=None) -> bool:
        """
        Checks whether a given coordination geometry is valid (exists) and whether the parameters are coherent with
        each other.

        Args:
            mp_symbol: The mp_symbol of the coordination geometry.
            IUPAC_symbol: The IUPAC_symbol of the coordination geometry.
            IUCr_symbol: The IUCr_symbol of the coordination geometry.
            name: The name of the coordination geometry.
            cn: The coordination of the coordination geometry.
        """
        if name is not None:
            raise NotImplementedError('is_a_valid_coordination_geometry not implemented for the name')
        if mp_symbol is None and IUPAC_symbol is None and (IUCr_symbol is None):
            raise SyntaxError('missing argument for is_a_valid_coordination_geometry : at least one of mp_symbol, IUPAC_symbol and IUCr_symbol must be passed to the function')
        if mp_symbol is not None:
            try:
                cg = self.get_geometry_from_mp_symbol(mp_symbol)
                if IUPAC_symbol is not None and IUPAC_symbol != cg.IUPAC_symbol:
                    return False
                if IUCr_symbol is not None and IUCr_symbol != cg.IUCr_symbol:
                    return False
                if cn is not None and int(cn) != int(cg.coordination_number):
                    return False
                return True
            except LookupError:
                return False
        elif IUPAC_symbol is not None:
            try:
                cg = self.get_geometry_from_IUPAC_symbol(IUPAC_symbol)
                if IUCr_symbol is not None and IUCr_symbol != cg.IUCr_symbol:
                    return False
                if cn is not None and cn != cg.coordination_number:
                    return False
                return True
            except LookupError:
                return False
        elif IUCr_symbol is not None:
            try:
                cg = self.get_geometry_from_IUCr_symbol(IUCr_symbol)
                if cn is not None and cn != cg.coordination_number:
                    return False
                return True
            except LookupError:
                return True
        raise RuntimeError('Should not be here!')

    def pretty_print(self, type='implemented_geometries', maxcn=8, additional_info=None):
        """
        Return a string with a list of the Coordination Geometries.

        Args:
            type: Type of string to be returned (all_geometries, all_geometries_latex_images, all_geometries_latex,
                implemented_geometries).
            maxcn: Maximum coordination.
            additional_info: Whether to add some additional info for each coordination geometry.

        Returns:
            str: description of the list of coordination geometries.
        """
        if type == 'all_geometries_latex_images':
            output = ''
            for cn in range(1, maxcn + 1):
                output += f'\\section*{{Coordination {cn}}}\n\n'
                for cg in self.get_implemented_geometries(coordination=cn, returned='cg'):
                    output += f'\\subsubsection*{{{cg.mp_symbol} : {cg.get_name()}}}\n\n'
                    output += f'IUPAC : {cg.IUPAC_symbol}\n\nIUCr : {cg.IUCr_symbol}\n\n'
                    output += '\\begin{center}\n'
                    output += f'\\includegraphics[scale=0.15]{{images/{cg.mp_symbol.split(':')[0]}_'
                    output += f'{cg.mp_symbol.split(':')[1]}.png}}\n'
                    output += '\\end{center}\n\n'
                for cg in self.get_not_implemented_geometries(cn, returned='cg'):
                    output += f'\\subsubsection*{{{cg.mp_symbol} : {cg.get_name()}}}\n\n'
                    output += f'IUPAC : {cg.IUPAC_symbol}\n\nIUCr : {cg.IUCr_symbol}\n\n'
        elif type == 'all_geometries_latex':
            output = ''
            for cn in range(1, maxcn + 1):
                output += f'\\subsection*{{Coordination {cn}}}\n\n'
                output += '\\begin{itemize}\n'
                for cg in self.get_implemented_geometries(coordination=cn, returned='cg'):
                    escaped_mp_symbol = cg.mp_symbol.replace('_', '\\_')
                    output += f'\\item {escaped_mp_symbol} $\\rightarrow$ {cg.get_name()} '
                    output += f'(IUPAC : {cg.IUPAC_symbol_str} - IUCr : '
                    output += f'{cg.IUCr_symbol_str.replace('[', '$[$').replace(']', '$]$')})\n'
                for cg in self.get_not_implemented_geometries(cn, returned='cg'):
                    escaped_mp_symbol = cg.mp_symbol.replace('_', '\\_')
                    output += f'\\item {escaped_mp_symbol} $\\rightarrow$ {cg.get_name()} '
                    output += f'(IUPAC : {cg.IUPAC_symbol_str} - IUCr : '
                    output += f'{cg.IUCr_symbol_str.replace('[', '$[$').replace(']', '$]$')})\n'
                output += '\\end{itemize}\n\n'
        else:
            output = '+-------------------------+\n| Coordination geometries |\n+-------------------------+\n\n'
            for cn in range(1, maxcn + 1):
                output += f'==>> CN = {cn} <<==\n'
                if type == 'implemented_geometries':
                    for cg in self.get_implemented_geometries(coordination=cn):
                        if additional_info is not None:
                            if 'nb_hints' in additional_info:
                                addinfo = ' *' if cg.neighbors_sets_hints is not None else ''
                            else:
                                addinfo = ''
                        else:
                            addinfo = ''
                        output += f' - {cg.mp_symbol} : {cg.get_name()}{addinfo}\n'
                elif type == 'all_geometries':
                    for cg in self.get_geometries(coordination=cn):
                        output += f' - {cg.mp_symbol} : {cg.get_name()}\n'
                output += '\n'
        return output