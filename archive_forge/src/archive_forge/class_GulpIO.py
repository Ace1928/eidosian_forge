from __future__ import annotations
import os
import re
import subprocess
from monty.tempfile import ScratchDir
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class GulpIO:
    """To generate GULP input and process output."""

    @staticmethod
    def keyword_line(*args):
        """Checks if the input args are proper gulp keywords and
        generates the 1st line of gulp input. Full keywords are expected.

        Args:
            args: 1st line keywords
        """
        gin = ' '.join(args)
        gin += '\n'
        return gin

    @staticmethod
    def structure_lines(structure: Structure, cell_flg: bool=True, frac_flg: bool=True, anion_shell_flg: bool=True, cation_shell_flg: bool=False, symm_flg: bool=True):
        """Generates GULP input string corresponding to pymatgen structure.

        Args:
            structure: pymatgen Structure object
            cell_flg (default = True): Option to use lattice parameters.
            frac_flg (default = True): If True, fractional coordinates
                are used. Else, Cartesian coordinates in Angstroms are used.
                ******
                GULP convention is to use fractional coordinates for periodic
                structures and Cartesian coordinates for non-periodic
                structures.
                ******
            anion_shell_flg (default = True): If True, anions are considered
                polarizable.
            cation_shell_flg (default = False): If True, cations are
                considered polarizable.
            symm_flg (default = True): If True, symmetry information is also
                written.

        Returns:
            string containing structure for GULP input
        """
        gin = ''
        if cell_flg:
            gin += 'cell\n'
            lattice = structure.lattice
            alpha, beta, gamma = lattice.angles
            a, b, c = lattice.lengths
            lat_str = f'{a:6f} {b:6f} {c:6f} {alpha:6f} {beta:6f} {gamma:6f}'
            gin += lat_str + '\n'
        if frac_flg:
            gin += 'frac\n'
            coords_key = 'frac_coords'
        else:
            gin += 'cart\n'
            coords_key = 'coords'
        for site in structure:
            coord = [str(i) for i in getattr(site, coords_key)]
            specie = site.specie
            core_site_desc = f'{specie.symbol} core {' '.join(coord)}\n'
            gin += core_site_desc
            if specie in _anions and anion_shell_flg or (specie in _cations and cation_shell_flg):
                shel_site_desc = f'{specie.symbol} shel {' '.join(coord)}\n'
                gin += shel_site_desc
            else:
                pass
        if symm_flg:
            gin += 'space\n'
            gin += str(SpacegroupAnalyzer(structure).get_space_group_number()) + '\n'
        return gin

    @staticmethod
    def specie_potential_lines(structure, potential, **kwargs):
        """Generates GULP input specie and potential string for pymatgen
        structure.

        Args:
            structure: pymatgen Structure object
            potential: String specifying the type of potential used
            kwargs: Additional parameters related to potential. For
                potential == "buckingham",
                anion_shell_flg (default = False):
                If True, anions are considered polarizable.
                anion_core_chrg=float
                anion_shell_chrg=float
                cation_shell_flg (default = False):
                If True, cations are considered polarizable.
                cation_core_chrg=float
                cation_shell_chrg=float

        Returns:
            string containing specie and potential specification for gulp
            input.
        """
        raise NotImplementedError('gulp_specie_potential not yet implemented.\nUse library_line instead')

    @staticmethod
    def library_line(file_name):
        """Specifies GULP library file to read species and potential parameters.
        If using library don't specify species and potential
        in the input file and vice versa. Make sure the elements of
        structure are in the library file.

        Args:
            file_name: Name of GULP library file

        Returns:
            GULP input string specifying library option
        """
        gulp_lib_set = 'GULP_LIB' in os.environ

        def readable(file):
            return os.path.isfile(file) and os.access(file, os.R_OK)
        gin = ''
        dirpath, _fname = os.path.split(file_name)
        if dirpath and readable(file_name):
            gin = f'library {file_name}'
        else:
            fpath = os.path.join(os.getcwd(), file_name)
            if readable(fpath):
                gin = f'library {fpath}'
            elif gulp_lib_set:
                fpath = os.path.join(os.environ['GULP_LIB'], file_name)
                if readable(fpath):
                    gin = f'library {file_name}'
        if gin:
            return gin + '\n'
        raise GulpError('GULP library not found')

    def buckingham_input(self, structure: Structure, keywords, library=None, uc=True, valence_dict=None):
        """Gets a GULP input for an oxide structure and buckingham potential
        from library.

        Args:
            structure: pymatgen Structure
            keywords: GULP first line keywords.
            library (Default=None): File containing the species and potential.
            uc (Default=True): Unit Cell Flag.
            valence_dict: {El: valence}
        """
        gin = self.keyword_line(*keywords)
        gin += self.structure_lines(structure, symm_flg=not uc)
        if not library:
            gin += self.buckingham_potential(structure, valence_dict)
        else:
            gin += self.library_line(library)
        return gin

    @staticmethod
    def buckingham_potential(structure, val_dict=None):
        """Generate species, buckingham, and spring options for an oxide structure
        using the parameters in default libraries.

        Ref:
            1. G.V. Lewis and C.R.A. Catlow, J. Phys. C: Solid State Phys.,
               18, 1149-1161 (1985)
            2. T.S.Bush, J.D.Gale, C.R.A.Catlow and P.D. Battle,
               J. Mater Chem., 4, 831-837 (1994)

        Args:
            structure: pymatgen Structure
            val_dict (Needed if structure is not charge neutral): {El:valence}
                dict, where El is element.
        """
        if not val_dict:
            try:
                el = [site.specie.symbol for site in structure]
                valences = [site.specie.oxi_state for site in structure]
                val_dict = dict(zip(el, valences))
            except AttributeError:
                bv = BVAnalyzer()
                el = [site.specie.symbol for site in structure]
                valences = bv.get_valences(structure)
                val_dict = dict(zip(el, valences))
        bpb = BuckinghamPotential('bush')
        bpl = BuckinghamPotential('lewis')
        gin = ''
        for key in val_dict:
            use_bush = True
            el = re.sub('[1-9,+,\\-]', '', key)
            if el not in bpb.species_dict or val_dict[key] != bpb.species_dict[el]['oxi']:
                use_bush = False
            if use_bush:
                gin += 'species \n'
                gin += bpb.species_dict[el]['inp_str']
                gin += 'buckingham \n'
                gin += bpb.pot_dict[el]
                gin += 'spring \n'
                gin += bpb.spring_dict[el]
                continue
            if el != 'O':
                k = f'{el}_{int(val_dict[key])}+'
                if k not in bpl.species_dict:
                    raise GulpError(f'Element {k} not in library')
                gin += 'species\n'
                gin += bpl.species_dict[k]
                gin += 'buckingham\n'
                gin += bpl.pot_dict[k]
            else:
                gin += 'species\n'
                k = 'O_core'
                gin += bpl.species_dict[k]
                k = 'O_shel'
                gin += bpl.species_dict[k]
                gin += 'buckingham\n'
                gin += bpl.pot_dict[key]
                gin += 'spring\n'
                gin += bpl.spring_dict[key]
        return gin

    def tersoff_input(self, structure: Structure, periodic=False, uc=True, *keywords):
        """Gets a GULP input with Tersoff potential for an oxide structure.

        Args:
            structure: pymatgen Structure
            periodic (Default=False): Flag denoting whether periodic
                boundary conditions are used
            library (Default=None): File containing the species and potential.
            uc (Default=True): Unit Cell Flag.
            keywords: GULP first line keywords.
        """
        gin = self.keyword_line(*keywords)
        gin += self.structure_lines(structure, cell_flg=periodic, frac_flg=periodic, anion_shell_flg=False, cation_shell_flg=False, symm_flg=not uc)
        gin += self.tersoff_potential(structure)
        return gin

    @staticmethod
    def tersoff_potential(structure):
        """Generate the species, Tersoff potential lines for an oxide structure.

        Args:
            structure: pymatgen Structure
        """
        bv = BVAnalyzer()
        el = [site.specie.symbol for site in structure]
        valences = bv.get_valences(structure)
        el_val_dict = dict(zip(el, valences))
        gin = 'species \n'
        qerf_str = 'qerfc\n'
        for key, value in el_val_dict.items():
            if key != 'O' and value % 1 != 0:
                raise SystemError('Oxide has mixed valence on metal')
            specie_str = f'{key} core {value}\n'
            gin += specie_str
            qerf_str += f'{key} {key} 0.6000 10.0000 \n'
        gin += '# noelectrostatics \n Morse \n'
        met_oxi_ters = TersoffPotential().data
        for key, value in el_val_dict.items():
            if key != 'O':
                metal = f'{key}({int(value)})'
                ters_pot_str = met_oxi_ters[metal]
                gin += ters_pot_str
        gin += qerf_str
        return gin

    @staticmethod
    def get_energy(gout: str):
        """
        Args:
            gout (str): GULP output string.

        Returns:
            Energy
        """
        energy = None
        for line in gout.split('\n'):
            if 'Total lattice energy' in line and 'eV' in line or ('Non-primitive unit cell' in line and 'eV' in line):
                energy = line.split()
        if energy:
            return float(energy[4])
        raise GulpError('Energy not found in Gulp output')

    @staticmethod
    def get_relaxed_structure(gout: str):
        """
        Args:
            gout (str): GULP output string.

        Returns:
            Structure: relaxed structure.
        """
        structure_lines = []
        cell_param_lines = []
        output_lines = gout.split('\n')
        n_lines = len(output_lines)
        idx = 0
        a = b = c = alpha = beta = gamma = 0.0
        while idx < n_lines:
            line = output_lines[idx]
            if 'Full cell parameters' in line:
                idx += 2
                line = output_lines[idx]
                a = float(line.split()[8])
                alpha = float(line.split()[11])
                line = output_lines[idx + 1]
                b = float(line.split()[8])
                beta = float(line.split()[11])
                line = output_lines[idx + 2]
                c = float(line.split()[8])
                gamma = float(line.split()[11])
                idx += 3
                break
            if 'Cell parameters' in line:
                idx += 2
                line = output_lines[idx]
                a = float(line.split()[2])
                alpha = float(line.split()[5])
                line = output_lines[idx + 1]
                b = float(line.split()[2])
                beta = float(line.split()[5])
                line = output_lines[idx + 2]
                c = float(line.split()[2])
                gamma = float(line.split()[5])
                idx += 3
                break
            idx += 1
        while idx < n_lines:
            line = output_lines[idx]
            if 'Final fractional coordinates of atoms' in line:
                idx += 6
                line = output_lines[idx]
                while line[0:2] != '--':
                    structure_lines.append(line)
                    idx += 1
                    line = output_lines[idx]
                idx += 9
                line = output_lines[idx]
                if 'Final cell parameters' in line:
                    idx += 3
                    for del_i in range(6):
                        line = output_lines[idx + del_i]
                        cell_param_lines.append(line)
                break
            idx += 1
        if structure_lines:
            sp = []
            coords = []
            for line in structure_lines:
                fields = line.split()
                if fields[2] == 'c':
                    sp.append(fields[1])
                    coords.append([float(x) for x in fields[3:6]])
        else:
            raise OSError('No structure found')
        if cell_param_lines:
            a = float(cell_param_lines[0].split()[1])
            b = float(cell_param_lines[1].split()[1])
            c = float(cell_param_lines[2].split()[1])
            alpha = float(cell_param_lines[3].split()[1])
            beta = float(cell_param_lines[4].split()[1])
            gamma = float(cell_param_lines[5].split()[1])
        if not all([a, b, c, alpha, beta, gamma]):
            raise ValueError(f'Missing lattice parameters in Gulp output: a={a!r}, b={b!r}, c={c!r}, alpha={alpha!r}, beta={beta!r}, gamma={gamma!r}')
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        return Structure(lattice, sp, coords)