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
class GaussianInput:
    """An object representing a Gaussian input file."""
    _zmat_patt = re.compile('^(\\w+)*([\\s,]+(\\w+)[\\s,]+(\\w+))*[\\-\\.\\s,\\w]*$')
    _xyz_patt = re.compile('^(\\w+)[\\s,]+([\\d\\.eE\\-]+)[\\s,]+([\\d\\.eE\\-]+)[\\s,]+([\\d\\.eE\\-]+)[\\-\\.\\s,\\w.]*$')

    def __init__(self, mol, charge=None, spin_multiplicity=None, title=None, functional='HF', basis_set='6-31G(d)', route_parameters=None, input_parameters=None, link0_parameters=None, dieze_tag='#P', gen_basis=None):
        """
        Args:
            mol: Input molecule. It can either be a Molecule object,
                a string giving the geometry in a format supported by Gaussian,
                or ``None``. If the molecule is ``None``, you will need to use
                read it in from a checkpoint. Consider adding ``CHK`` to the
                ``link0_parameters``.
            charge: Charge of the molecule. If None, charge on molecule is used.
                Defaults to None. This allows the input file to be set a
                charge independently from the molecule itself.
                If ``mol`` is not a Molecule object, then you must specify a charge.
            spin_multiplicity: Spin multiplicity of molecule. Defaults to None,
                which means that the spin multiplicity is set to 1 if the
                molecule has no unpaired electrons and to 2 if there are
                unpaired electrons. If ``mol`` is not a Molecule object, then you
                 must specify the multiplicity
            title: Title for run. Defaults to formula of molecule if None.
            functional: Functional for run.
            basis_set: Basis set for run.
            route_parameters: Additional route parameters as a dict. For example,
                {'SP':"", "SCF":"Tight"}
            input_parameters: Additional input parameters for run as a dict. Used
                for example, in PCM calculations. E.g., {"EPS":12}
            link0_parameters: Link0 parameters as a dict. E.g., {"%mem": "1000MW"}
            dieze_tag: # preceding the route line. E.g. "#p"
            gen_basis: allows a user-specified basis set to be used in a Gaussian
                calculation. If this is not None, the attribute ``basis_set`` will
                be set to "Gen".
        """
        self._mol = mol
        if isinstance(mol, Molecule):
            self.charge = charge if charge is not None else mol.charge
            n_electrons = mol.charge + mol.nelectrons - self.charge
            if spin_multiplicity is not None:
                self.spin_multiplicity = spin_multiplicity
                if (n_electrons + spin_multiplicity) % 2 != 1:
                    raise ValueError(f'Charge of {self.charge} and spin multiplicity of {spin_multiplicity} is not possible for this molecule')
            else:
                self.spin_multiplicity = 1 if n_electrons % 2 == 0 else 2
            self.title = title or self._mol.formula
        else:
            self.charge = charge
            self.spin_multiplicity = spin_multiplicity
            self.title = title or 'Restart'
        self.functional = functional
        self.basis_set = basis_set
        self.link0_parameters = link0_parameters or {}
        self.route_parameters = route_parameters or {}
        self.input_parameters = input_parameters or {}
        self.dieze_tag = dieze_tag if dieze_tag[0] == '#' else f'#{dieze_tag}'
        self.gen_basis = gen_basis
        if gen_basis is not None:
            self.basis_set = 'Gen'

    @property
    def molecule(self):
        """Returns molecule associated with this GaussianInput."""
        return self._mol

    @staticmethod
    def _parse_coords(coord_lines):
        """Helper method to parse coordinates."""
        paras = {}
        var_pattern = re.compile('^([A-Za-z]+\\S*)[\\s=,]+([\\d\\-\\.]+)$')
        for line in coord_lines:
            m = var_pattern.match(line.strip())
            if m:
                paras[m.group(1).strip('=')] = float(m.group(2))
        species = []
        coords = []
        zmode = False
        for line in coord_lines:
            line = line.strip()
            if not line:
                break
            if not zmode and GaussianInput._xyz_patt.match(line):
                m = GaussianInput._xyz_patt.match(line)
                species.append(m.group(1))
                tokens = re.split('[,\\s]+', line.strip())
                if len(tokens) > 4:
                    coords.append([float(i) for i in tokens[2:5]])
                else:
                    coords.append([float(i) for i in tokens[1:4]])
            elif GaussianInput._zmat_patt.match(line):
                zmode = True
                tokens = re.split('[,\\s]+', line.strip())
                species.append(tokens[0])
                tokens.pop(0)
                if len(tokens) == 0:
                    coords.append(np.array([0, 0, 0]))
                else:
                    nn = []
                    parameters = []
                    while len(tokens) > 1:
                        ind = tokens.pop(0)
                        data = tokens.pop(0)
                        try:
                            nn.append(int(ind))
                        except ValueError:
                            nn.append(species.index(ind) + 1)
                        try:
                            val = float(data)
                            parameters.append(val)
                        except ValueError:
                            if data.startswith('-'):
                                parameters.append(-paras[data[1:]])
                            else:
                                parameters.append(paras[data])
                    if len(nn) == 1:
                        coords.append(np.array([0, 0, parameters[0]]))
                    elif len(nn) == 2:
                        coords1 = coords[nn[0] - 1]
                        coords2 = coords[nn[1] - 1]
                        bl = parameters[0]
                        angle = parameters[1]
                        axis = [0, 1, 0]
                        op = SymmOp.from_origin_axis_angle(coords1, axis, angle)
                        coord = op.operate(coords2)
                        vec = coord - coords1
                        coord = vec * bl / np.linalg.norm(vec) + coords1
                        coords.append(coord)
                    elif len(nn) == 3:
                        coords1 = coords[nn[0] - 1]
                        coords2 = coords[nn[1] - 1]
                        coords3 = coords[nn[2] - 1]
                        bl = parameters[0]
                        angle = parameters[1]
                        dih = parameters[2]
                        v1 = coords3 - coords2
                        v2 = coords1 - coords2
                        axis = np.cross(v1, v2)
                        op = SymmOp.from_origin_axis_angle(coords1, axis, angle)
                        coord = op.operate(coords2)
                        v1 = coord - coords1
                        v2 = coords1 - coords2
                        v3 = np.cross(v1, v2)
                        adj = get_angle(v3, axis)
                        axis = coords1 - coords2
                        op = SymmOp.from_origin_axis_angle(coords1, axis, dih - adj)
                        coord = op.operate(coord)
                        vec = coord - coords1
                        coord = vec * bl / np.linalg.norm(vec) + coords1
                        coords.append(coord)

        def _parse_species(sp_str):
            """
            The species specification can take many forms. E.g.,
            simple integers representing atomic numbers ("8"),
            actual species string ("C") or a labelled species ("C1").
            Sometimes, the species string is also not properly capitalized,
            e.g, ("c1"). This method should take care of these known formats.
            """
            try:
                return int(sp_str)
            except ValueError:
                sp = re.sub('\\d', '', sp_str)
                return sp.capitalize()
        species = [_parse_species(sp) for sp in species]
        return Molecule(species, coords)

    @classmethod
    def from_str(cls, contents: str) -> Self:
        """
        Creates GaussianInput from a string.

        Args:
            contents: String representing an Gaussian input file.

        Returns:
            GaussianInput object
        """
        lines = [line.strip() for line in contents.split('\n')]
        link0_patt = re.compile('^(%.+)\\s*=\\s*(.+)')
        link0_dict = {}
        for line in lines:
            if link0_patt.match(line):
                m = link0_patt.match(line)
                assert m is not None
                link0_dict[m.group(1).strip('=')] = m.group(2)
        route_patt = re.compile('^#[sSpPnN]*.*')
        route = ''
        route_index = None
        for idx, line in enumerate(lines):
            if route_patt.match(line):
                route += f' {line}'
                route_index = idx
            elif (line == '' or line.isspace()) and route_index:
                break
            if route_index:
                route += f' {line}'
                route_index = idx
        functional, basis_set, route_paras, dieze_tag = read_route_line(route)
        ind = 2
        title = []
        assert route_index is not None, 'route_index cannot be None'
        while lines[route_index + ind].strip():
            title.append(lines[route_index + ind].strip())
            ind += 1
        title_str = ' '.join(title)
        ind += 1
        tokens = re.split('[,\\s]+', lines[route_index + ind])
        charge = int(float(tokens[0]))
        spin_mult = int(tokens[1])
        coord_lines = []
        spaces = 0
        input_paras = {}
        ind += 1
        if cls._xyz_patt.match(lines[route_index + ind]):
            spaces += 1
        for i in range(route_index + ind, len(lines)):
            if lines[i].strip() == '':
                spaces += 1
            if spaces >= 2:
                d = lines[i].split('=')
                if len(d) == 2:
                    input_paras[d[0]] = d[1]
            else:
                coord_lines.append(lines[i].strip())
        mol = cls._parse_coords(coord_lines)
        mol.set_charge_and_spin(charge, spin_mult)
        return cls(mol, charge=charge, spin_multiplicity=spin_mult, title=title_str, functional=functional, basis_set=basis_set, route_parameters=route_paras, input_parameters=input_paras, link0_parameters=link0_dict, dieze_tag=dieze_tag)

    @classmethod
    def from_file(cls, filename: str | Path) -> Self:
        """
        Creates GaussianInput from a file.

        Args:
            filename: Gaussian input filename

        Returns:
            GaussianInput object
        """
        with zopen(filename, mode='r') as file:
            return cls.from_str(file.read())

    def get_zmatrix(self):
        """Returns a z-matrix representation of the molecule."""
        return self._mol.get_zmatrix()

    def get_cart_coords(self) -> str:
        """Return the Cartesian coordinates of the molecule."""
        outs = []
        for site in self._mol:
            outs.append(f'{site.species_string} {' '.join((f'{x:0.6f}' for x in site.coords))}')
        return '\n'.join(outs)

    def __str__(self):
        return self.to_str()

    def to_str(self, cart_coords=False):
        """Return GaussianInput string.

        Args:
            cart_coords (bool): If True, return Cartesian coordinates instead of z-matrix.
                Defaults to False.
        """

        def para_dict_to_str(para, joiner=' '):
            para_str = []
            for par, val in sorted(para.items()):
                if val is None or val == '':
                    para_str.append(par)
                elif isinstance(val, dict):
                    val_str = para_dict_to_str(val, joiner=',')
                    para_str.append(f'{par}=({val_str})')
                else:
                    para_str.append(f'{par}={val}')
            return joiner.join(para_str)
        output = []
        if self.link0_parameters:
            output.append(para_dict_to_str(self.link0_parameters, '\n'))
        func_str = '' if self.functional is None else self.functional.strip()
        bset_str = '' if self.basis_set is None else self.basis_set.strip()
        if func_str != '' and bset_str != '':
            func_bset_str = f' {func_str}/{bset_str}'
        else:
            func_bset_str = f' {func_str}{bset_str}'.rstrip()
        output += (f'{self.dieze_tag}{func_bset_str} {para_dict_to_str(self.route_parameters)}', '', self.title, '')
        charge_str = '' if self.charge is None else f'{self.charge:.0f}'
        multip_str = '' if self.spin_multiplicity is None else f' {self.spin_multiplicity:.0f}'
        output.append(f'{charge_str}{multip_str}')
        if isinstance(self._mol, Molecule):
            if cart_coords is True:
                output.append(self.get_cart_coords())
            else:
                output.append(self.get_zmatrix())
        elif self._mol is not None:
            output.append(str(self._mol))
        output.append('')
        if self.gen_basis is not None:
            output.append(f'{self.gen_basis}\n')
        output.extend((para_dict_to_str(self.input_parameters, '\n'), '\n'))
        return '\n'.join(output)

    def write_file(self, filename, cart_coords=False):
        """
        Write the input string into a file.

        Option: see __str__ method
        """
        with zopen(filename, mode='w') as file:
            file.write(self.to_str(cart_coords))

    def as_dict(self):
        """MSONable dict"""
        return {'@module': type(self).__module__, '@class': type(self).__name__, 'molecule': self.molecule.as_dict(), 'functional': self.functional, 'basis_set': self.basis_set, 'route_parameters': self.route_parameters, 'title': self.title, 'charge': self.charge, 'spin_multiplicity': self.spin_multiplicity, 'input_parameters': self.input_parameters, 'link0_parameters': self.link0_parameters, 'dieze_tag': self.dieze_tag}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """
        Args:
            dct: dict

        Returns:
            GaussianInput
        """
        return cls(mol=Molecule.from_dict(dct['molecule']), functional=dct['functional'], basis_set=dct['basis_set'], route_parameters=dct['route_parameters'], title=dct['title'], charge=dct['charge'], spin_multiplicity=dct['spin_multiplicity'], input_parameters=dct['input_parameters'], link0_parameters=dct['link0_parameters'])