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
class GaussianOutput:
    """
    Parser for Gaussian output files.

    Note: Still in early beta.

    Attributes:
        structures (list[Structure]): All structures from the calculation in the standard orientation. If the
            symmetry is not considered, the standard orientation is not printed out
            and the input orientation is used instead. Check the `standard_orientation`
            attribute.
        structures_input_orientation (list): All structures from the calculation in the input
            orientation or the Z-matrix orientation (if an opt=z-matrix was requested).
        opt_structures (list): All optimized structures from the calculation in the standard
            orientation, if the attribute 'standard_orientation' is True, otherwise in the input
            or the Z-matrix orientation.
        energies (list): All energies from the calculation.
        eigenvalues (list): List of eigenvalues for the last geometry.
        MO_coefficients (list): Matrix of MO coefficients for the last geometry.
        cart_forces (list): All Cartesian forces from the calculation.
        frequencies (list): A list for each freq calculation and for each mode of a dict with
            {
                "frequency": freq in cm-1,
                "symmetry": symmetry tag
                "r_mass": Reduce mass,
                "f_constant": force constant,
                "IR_intensity": IR Intensity,
                "mode": normal mode
             }
            The normal mode is a 1D vector of dx, dy dz of each atom.
        hessian (ndarray): Matrix of second derivatives of the energy with respect to cartesian
            coordinates in the input orientation frame. Need #P in the route section in order to
            be in the output.
        properly_terminated (bool): True if run has properly terminated.
        is_pcm (bool): True if run is a PCM run.
        is_spin (bool): True if it is an unrestricted run.
        stationary_type (str): If it is a relaxation run, indicates whether it is a minimum
            (Minimum) or a saddle point ("Saddle").
        corrections (dict): Thermochemical corrections if this run is a Freq run as a dict. Keys
            are "Zero-point", "Thermal", "Enthalpy" and "Gibbs Free Energy".
        functional (str): Functional used in the run.
        basis_set (str): Basis set used in the run.
        route (dict): Additional route parameters as a dict. For example,
            {'SP':"", "SCF":"Tight"}.
        dieze_tag (str): # preceding the route line, e.g. "#P".
        link0 (dict): Link0 parameters as a dict. E.g., {"%mem": "1000MW"}.
        charge (int): Charge for structure.
        spin_multiplicity (int): Spin multiplicity for structure.
        num_basis_func (int): Number of basis functions in the run.
        electrons (tuple): Number of alpha and beta electrons as (N alpha, N beta).
        pcm (dict): PCM parameters and output if available.
        errors (list): Error if not properly terminated (list to be completed in error_defs).
        Mulliken_charges (list): Mulliken atomic charges.
        eigenvectors (dict): Matrix of shape (num_basis_func, num_basis_func). Each column is an
            eigenvectors and contains AO coefficients of an MO.
            eigenvectors[Spin] = mat(num_basis_func, num_basis_func).
        molecular_orbital (dict): MO development coefficients on AO in a more convenient array dict
            for each atom and basis set label.
            mo[Spin][OM j][atom i] = {AO_k: coeff, AO_k: coeff ... }.
        atom_basis_labels (list): Labels of AO for each atoms. These labels are those used in the
            output of molecular orbital coefficients (POP=Full) and in the molecular_orbital array
            dict. atom_basis_labels[iatom] = [AO_k, AO_k, ...].
        resumes (list): List of gaussian data resume given at the end of the output file before
            the quotation. The resumes are given as string.
        title (str): Title of the gaussian run.
        standard_orientation (bool): If True, the geometries stored in the structures are in the
            standard orientation. Else, the geometries are in the input orientation.
        bond_orders (dict): Dict of bond order values read in the output file such as:
            {(0, 1): 0.8709, (1, 6): 1.234, ...}.
            The keys are the atom indexes and the values are the Wiberg bond indexes that are
            printed using `pop=NBOREAD` and `$nbo bndidx $end`.

    Methods:
        to_input()
            Return a GaussianInput object using the last geometry and the same
            calculation parameters.

        read_scan()
            Read a potential energy surface from a gaussian scan calculation.

        get_scan_plot()
            Get a matplotlib plot of the potential energy surface

        save_scan_plot()
            Save a matplotlib plot of the potential energy surface to a file
    """

    def __init__(self, filename):
        """
        Args:
            filename: Filename of Gaussian output file.
        """
        self.filename = filename
        self._parse(filename)

    @property
    def final_energy(self):
        """Final energy in Gaussian output."""
        return self.energies[-1]

    @property
    def final_structure(self):
        """Final structure in Gaussian output."""
        return self.structures[-1]

    def _parse(self, filename):
        start_patt = re.compile(' \\(Enter \\S+l101\\.exe\\)')
        route_patt = re.compile(' #[pPnNtT]*.*')
        link0_patt = re.compile('^\\s(%.+)\\s*=\\s*(.+)')
        charge_mul_patt = re.compile('Charge\\s+=\\s*([-\\d]+)\\s+Multiplicity\\s+=\\s*(\\d+)')
        num_basis_func_patt = re.compile('([0-9]+)\\s+basis functions')
        num_elec_patt = re.compile('(\\d+)\\s+alpha electrons\\s+(\\d+)\\s+beta electrons')
        pcm_patt = re.compile('Polarizable Continuum Model')
        stat_type_patt = re.compile('imaginary frequencies')
        scf_patt = re.compile('E\\(.*\\)\\s*=\\s*([-\\.\\d]+)\\s+')
        mp2_patt = re.compile('EUMP2\\s*=\\s*(.*)')
        oniom_patt = re.compile('ONIOM:\\s+extrapolated energy\\s*=\\s*(.*)')
        termination_patt = re.compile('(Normal|Error) termination')
        error_patt = re.compile('(! Non-Optimized Parameters !|Convergence failure)')
        mulliken_patt = re.compile('^\\s*(Mulliken charges|Mulliken atomic charges)')
        mulliken_charge_patt = re.compile('^\\s+(\\d+)\\s+([A-Z][a-z]?)\\s*(\\S*)')
        end_mulliken_patt = re.compile('(Sum of Mulliken )(.*)(charges)\\s*=\\s*(\\D)')
        std_orientation_patt = re.compile('Standard orientation')
        input_orientation_patt = re.compile('Input orientation|Z-Matrix orientation')
        orbital_patt = re.compile('(Alpha|Beta)\\s*\\S+\\s*eigenvalues --(.*)')
        thermo_patt = re.compile('(Zero-point|Thermal) correction(.*)=\\s+([\\d\\.-]+)')
        forces_on_patt = re.compile('Center\\s+Atomic\\s+Forces\\s+\\(Hartrees/Bohr\\)')
        forces_off_patt = re.compile('Cartesian\\s+Forces:\\s+Max.*RMS.*')
        forces_patt = re.compile('\\s+(\\d+)\\s+(\\d+)\\s+([0-9\\.-]+)\\s+([0-9\\.-]+)\\s+([0-9\\.-]+)')
        freq_on_patt = re.compile('Harmonic\\sfrequencies\\s+\\(cm\\*\\*-1\\),\\sIR\\sintensities.*Raman.*')
        normal_mode_patt = re.compile('\\s+(\\d+)\\s+(\\d+)\\s+([0-9\\.-]{4,5})\\s+([0-9\\.-]{4,5}).*')
        mo_coeff_patt = re.compile('Molecular Orbital Coefficients:')
        mo_coeff_name_patt = re.compile('\\d+\\s((\\d+|\\s+)\\s+([a-zA-Z]{1,2}|\\s+))\\s+(\\d+\\S+)')
        hessian_patt = re.compile('Force constants in Cartesian coordinates:')
        resume_patt = re.compile('^\\s1\\\\1\\\\GINC-\\S*')
        resume_end_patt = re.compile('^\\s.*\\\\\\\\@')
        bond_order_patt = re.compile('Wiberg bond index matrix in the NAO basis:')
        self.properly_terminated = False
        self.is_pcm = False
        self.stationary_type = 'Minimum'
        self.corrections = {}
        self.energies = []
        self.pcm = None
        self.errors = []
        self.Mulliken_charges = {}
        self.link0 = {}
        self.cart_forces = []
        self.frequencies = []
        self.eigenvalues = []
        self.is_spin = False
        self.hessian = None
        self.resumes = []
        self.title = None
        self.bond_orders = {}
        read_coord = 0
        read_mulliken = False
        read_eigen = False
        eigen_txt = []
        parse_stage = 0
        num_basis_found = False
        terminated = False
        parse_forces = False
        forces = []
        parse_freq = False
        frequencies = []
        read_mo = False
        parse_hessian = False
        route_line = ''
        standard_orientation = False
        parse_bond_order = False
        input_structures = []
        std_structures = []
        geom_orientation = None
        opt_structures = []
        with zopen(filename, mode='rt') as file:
            for line in file:
                if parse_stage == 0:
                    if start_patt.search(line):
                        parse_stage = 1
                    elif link0_patt.match(line):
                        m = link0_patt.match(line)
                        self.link0[m.group(1)] = m.group(2)
                    elif route_patt.search(line) or route_line != '':
                        if set(line.strip()) == {'-'}:
                            params = read_route_line(route_line)
                            self.functional = params[0]
                            self.basis_set = params[1]
                            self.route_parameters = params[2]
                            route_lower = {k.lower(): v for k, v in self.route_parameters.items()}
                            self.dieze_tag = params[3]
                            parse_stage = 1
                        else:
                            line = line.replace(' ', '', 1).rstrip('\n')
                            route_line += line
                elif parse_stage == 1:
                    if set(line.strip()) == {'-'} and self.title is None:
                        self.title = ''
                    elif self.title == '':
                        self.title = line.strip()
                    elif charge_mul_patt.search(line):
                        m = charge_mul_patt.search(line)
                        self.charge = int(m.group(1))
                        self.spin_multiplicity = int(m.group(2))
                        parse_stage = 2
                elif parse_stage == 2:
                    if self.is_pcm:
                        self._check_pcm(line)
                    if 'freq' in route_lower and thermo_patt.search(line):
                        m = thermo_patt.search(line)
                        if m.group(1) == 'Zero-point':
                            self.corrections['Zero-point'] = float(m.group(3))
                        else:
                            key = m.group(2).replace(' to ', '')
                            self.corrections[key] = float(m.group(3))
                    if read_coord:
                        [file.readline() for i in range(3)]
                        line = file.readline()
                        sp = []
                        coords = []
                        while set(line.strip()) != {'-'}:
                            tokens = line.split()
                            sp.append(Element.from_Z(int(tokens[1])))
                            coords.append([float(x) for x in tokens[3:6]])
                            line = file.readline()
                        read_coord = False
                        if geom_orientation == 'input':
                            input_structures.append(Molecule(sp, coords))
                        elif geom_orientation == 'standard':
                            std_structures.append(Molecule(sp, coords))
                    if parse_forces:
                        if (m := forces_patt.search(line)):
                            forces.extend([float(_v) for _v in m.groups()[2:5]])
                        elif forces_off_patt.search(line):
                            self.cart_forces.append(forces)
                            forces = []
                            parse_forces = False
                    if read_eigen:
                        if (m := orbital_patt.search(line)):
                            eigen_txt.append(line)
                        else:
                            read_eigen = False
                            self.eigenvalues = {Spin.up: []}
                            for eigen_line in eigen_txt:
                                if 'Alpha' in eigen_line:
                                    self.eigenvalues[Spin.up] += [float(e) for e in float_patt.findall(eigen_line)]
                                elif 'Beta' in eigen_line:
                                    if Spin.down not in self.eigenvalues:
                                        self.eigenvalues[Spin.down] = []
                                    self.eigenvalues[Spin.down] += [float(e) for e in float_patt.findall(eigen_line)]
                            eigen_txt = []
                    if not num_basis_found and num_basis_func_patt.search(line):
                        m = num_basis_func_patt.search(line)
                        self.num_basis_func = int(m.group(1))
                        num_basis_found = True
                    elif read_mo:
                        all_spin = [Spin.up]
                        if self.is_spin:
                            all_spin.append(Spin.down)
                        mat_mo = {}
                        for spin in all_spin:
                            mat_mo[spin] = np.zeros((self.num_basis_func, self.num_basis_func))
                            nMO = 0
                            end_mo = False
                            while nMO < self.num_basis_func and (not end_mo):
                                file.readline()
                                file.readline()
                                self.atom_basis_labels = []
                                for idx in range(self.num_basis_func):
                                    line = file.readline()
                                    m = mo_coeff_name_patt.search(line)
                                    if m.group(1).strip() != '':
                                        atom_idx = int(m.group(2)) - 1
                                        self.atom_basis_labels.append([m.group(4)])
                                    else:
                                        self.atom_basis_labels[atom_idx].append(m.group(4))
                                    coeffs = [float(c) for c in float_patt.findall(line)]
                                    for j, c in enumerate(coeffs):
                                        mat_mo[spin][idx, nMO + j] = c
                                nMO += len(coeffs)
                                line = file.readline()
                                if nMO < self.num_basis_func and ('Density Matrix:' in line or mo_coeff_patt.search(line)):
                                    end_mo = True
                                    warnings.warn('POP=regular case, matrix coefficients not complete')
                            file.readline()
                        self.eigenvectors = mat_mo
                        read_mo = False
                        mo = {}
                        for spin in all_spin:
                            mo[spin] = [[{} for iat in range(len(self.atom_basis_labels))] for j in range(self.num_basis_func)]
                            for j in range(self.num_basis_func):
                                idx = 0
                                for atom_idx, labels in enumerate(self.atom_basis_labels):
                                    for label in labels:
                                        mo[spin][j][atom_idx][label] = self.eigenvectors[spin][idx, j]
                                        idx += 1
                        self.molecular_orbital = mo
                    elif parse_freq:
                        while line.strip() != '':
                            ifreqs = [int(val) - 1 for val in line.split()]
                            for _ in ifreqs:
                                frequencies.append({'frequency': None, 'r_mass': None, 'f_constant': None, 'IR_intensity': None, 'symmetry': None, 'mode': []})
                            while 'Atom  AN' not in line:
                                if 'Frequencies --' in line:
                                    freqs = map(float, float_patt.findall(line))
                                    for ifreq, freq in zip(ifreqs, freqs):
                                        frequencies[ifreq]['frequency'] = freq
                                elif 'Red. masses --' in line:
                                    r_masses = map(float, float_patt.findall(line))
                                    for ifreq, r_mass in zip(ifreqs, r_masses):
                                        frequencies[ifreq]['r_mass'] = r_mass
                                elif 'Frc consts  --' in line:
                                    f_consts = map(float, float_patt.findall(line))
                                    for ifreq, f_const in zip(ifreqs, f_consts):
                                        frequencies[ifreq]['f_constant'] = f_const
                                elif 'IR Inten    --' in line:
                                    IR_intens = map(float, float_patt.findall(line))
                                    for ifreq, intens in zip(ifreqs, IR_intens):
                                        frequencies[ifreq]['IR_intensity'] = intens
                                else:
                                    syms = line.split()[:3]
                                    for ifreq, sym in zip(ifreqs, syms):
                                        frequencies[ifreq]['symmetry'] = sym
                                line = file.readline()
                            line = file.readline()
                            while normal_mode_patt.search(line):
                                values = list(map(float, float_patt.findall(line)))
                                for idx, ifreq in zip(range(0, len(values), 3), ifreqs):
                                    frequencies[ifreq]['mode'].extend(values[idx:idx + 3])
                                line = file.readline()
                        parse_freq = False
                        self.frequencies.append(frequencies)
                        frequencies = []
                    elif parse_hessian:
                        if not (input_structures or std_structures):
                            raise ValueError('Both input_structures and std_structures are empty.')
                        parse_hessian = False
                        self._parse_hessian(file, (input_structures or std_structures)[0])
                    elif parse_bond_order:
                        line = file.readline()
                        line = file.readline()
                        n_atoms = len(input_structures[0])
                        matrix = []
                        for _ in range(n_atoms):
                            line = file.readline()
                            matrix.append([float(v) for v in line.split()[2:]])
                        self.bond_orders = {}
                        for atom_idx in range(n_atoms):
                            for atom_jdx in range(atom_idx + 1, n_atoms):
                                self.bond_orders[atom_idx, atom_jdx] = matrix[atom_idx][atom_jdx]
                        parse_bond_order = False
                    elif termination_patt.search(line):
                        m = termination_patt.search(line)
                        if m.group(1) == 'Normal':
                            self.properly_terminated = True
                            terminated = True
                    elif error_patt.search(line):
                        error_defs = {'! Non-Optimized Parameters !': 'Optimization error', 'Convergence failure': 'SCF convergence error'}
                        m = error_patt.search(line)
                        self.errors.append(error_defs[m.group(1)])
                    elif num_elec_patt.search(line):
                        m = num_elec_patt.search(line)
                        self.electrons = (int(m.group(1)), int(m.group(2)))
                    elif not self.is_pcm and pcm_patt.search(line):
                        self.is_pcm = True
                        self.pcm = {}
                    elif 'freq' in route_lower and 'opt' in route_lower and stat_type_patt.search(line):
                        self.stationary_type = 'Saddle'
                    elif mp2_patt.search(line):
                        m = mp2_patt.search(line)
                        self.energies.append(float(m.group(1).replace('D', 'E')))
                    elif oniom_patt.search(line):
                        m = oniom_patt.matcher(line)
                        self.energies.append(float(m.group(1)))
                    elif scf_patt.search(line):
                        m = scf_patt.search(line)
                        self.energies.append(float(m.group(1)))
                    elif std_orientation_patt.search(line):
                        standard_orientation = True
                        geom_orientation = 'standard'
                        read_coord = True
                    elif input_orientation_patt.search(line):
                        geom_orientation = 'input'
                        read_coord = True
                    elif 'Optimization completed.' in line:
                        line = file.readline()
                        if ' -- Stationary point found.' not in line:
                            warnings.warn(f'\n{self.filename}: Optimization complete but this is not a stationary point')
                        if standard_orientation:
                            opt_structures.append(std_structures[-1])
                        else:
                            opt_structures.append(input_structures[-1])
                    elif not read_eigen and orbital_patt.search(line):
                        eigen_txt.append(line)
                        read_eigen = True
                    elif mulliken_patt.search(line):
                        mulliken_txt = []
                        read_mulliken = True
                    elif not parse_forces and forces_on_patt.search(line):
                        parse_forces = True
                    elif freq_on_patt.search(line):
                        parse_freq = True
                        [file.readline() for i in range(3)]
                    elif mo_coeff_patt.search(line):
                        if 'Alpha' in line:
                            self.is_spin = True
                        read_mo = True
                    elif hessian_patt.search(line):
                        parse_hessian = True
                    elif resume_patt.search(line):
                        resume = []
                        while not resume_end_patt.search(line):
                            resume.append(line)
                            line = file.readline()
                            if line == '\n':
                                break
                        resume.append(line)
                        resume = ''.join((r.strip() for r in resume))
                        self.resumes.append(resume)
                    elif bond_order_patt.search(line):
                        parse_bond_order = True
                    if read_mulliken:
                        if not end_mulliken_patt.search(line):
                            mulliken_txt.append(line)
                        else:
                            m = end_mulliken_patt.search(line)
                            mulliken_charges = {}
                            for line in mulliken_txt:
                                if mulliken_charge_patt.search(line):
                                    m = mulliken_charge_patt.search(line)
                                    dic = {int(m.group(1)): [m.group(2), float(m.group(3))]}
                                    mulliken_charges.update(dic)
                            read_mulliken = False
                            self.Mulliken_charges = mulliken_charges
        self.structures_input_orientation = input_structures
        if standard_orientation:
            self.structures = std_structures
        else:
            self.structures = input_structures
        self.opt_structures = opt_structures
        if not terminated:
            warnings.warn(f'\n{self.filename}: Termination error or bad Gaussian output file !')

    def _parse_hessian(self, file, structure):
        """
        Parse the hessian matrix in the output file.

        Args:
            file: file object
            structure: structure in the output file
        """
        ndf = 3 * len(structure)
        self.hessian = np.zeros((ndf, ndf))
        j_indices = range(5)
        ndf_idx = 0
        while ndf_idx < ndf:
            for i in range(ndf_idx, ndf):
                line = file.readline()
                vals = re.findall('\\s*([+-]?\\d+\\.\\d+[eEdD]?[+-]\\d+)', line)
                vals = [float(val.replace('D', 'E')) for val in vals]
                for val_idx, val in enumerate(vals):
                    j = j_indices[val_idx]
                    self.hessian[i, j] = val
                    self.hessian[j, i] = val
            ndf_idx += len(vals)
            line = file.readline()
            j_indices = [j + 5 for j in j_indices]

    def _check_pcm(self, line):
        energy_patt = re.compile('(Dispersion|Cavitation|Repulsion) energy\\s+\\S+\\s+=\\s+(\\S*)')
        total_patt = re.compile('with all non electrostatic terms\\s+\\S+\\s+=\\s+(\\S*)')
        parameter_patt = re.compile('(Eps|Numeral density|RSolv|Eps\\(inf[inity]*\\))\\s+=\\s*(\\S*)')
        if energy_patt.search(line):
            m = energy_patt.search(line)
            self.pcm[f'{m.group(1)} energy'] = float(m.group(2))
        elif total_patt.search(line):
            m = total_patt.search(line)
            self.pcm['Total energy'] = float(m.group(1))
        elif parameter_patt.search(line):
            m = parameter_patt.search(line)
            self.pcm[m.group(1)] = float(m.group(2))

    def as_dict(self):
        """JSON-serializable dict representation."""
        structure = self.final_structure
        dct = {'has_gaussian_completed': self.properly_terminated, 'nsites': len(structure)}
        comp = structure.composition
        dct['unit_cell_formula'] = comp.as_dict()
        dct['reduced_cell_formula'] = Composition(comp.reduced_formula).as_dict()
        dct['pretty_formula'] = comp.reduced_formula
        dct['is_pcm'] = self.is_pcm
        dct['errors'] = self.errors
        dct['Mulliken_charges'] = self.Mulliken_charges
        unique_symbols = sorted(dct['unit_cell_formula'])
        dct['elements'] = unique_symbols
        dct['nelements'] = len(unique_symbols)
        dct['charge'] = self.charge
        dct['spin_multiplicity'] = self.spin_multiplicity
        vin = {'route': self.route_parameters, 'functional': self.functional, 'basis_set': self.basis_set, 'nbasisfunctions': self.num_basis_func, 'pcm_parameters': self.pcm}
        dct['input'] = vin
        n_sites = len(self.final_structure)
        vout = {'energies': self.energies, 'final_energy': self.final_energy, 'final_energy_per_atom': self.final_energy / n_sites, 'molecule': structure.as_dict(), 'stationary_type': self.stationary_type, 'corrections': self.corrections}
        dct['output'] = vout
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        return dct

    def read_scan(self):
        """
        Read a potential energy surface from a gaussian scan calculation.

        Returns:
            dict[str, list]: {"energies": [...], "coords": {"d1": [...], "A2", [...], ... }}
            "energies" are the energies of all points of the potential energy
            surface. "coords" are the internal coordinates used to compute the
            potential energy surface and the internal coordinates optimized,
            labelled by their name as defined in the calculation.
        """
        scan_patt = re.compile('^\\sSummary of the potential surface scan:')
        optscan_patt = re.compile('^\\sSummary of Optimized Potential Surface Scan')
        coord_patt = re.compile('^\\s*(\\w+)((\\s*[+-]?\\d+\\.\\d+)+)')
        data = {'energies': [], 'coords': {}}
        with zopen(self.filename, mode='r') as file:
            line = file.readline()
            while line != '':
                if optscan_patt.match(line):
                    file.readline()
                    line = file.readline()
                    endScan = False
                    while not endScan:
                        data['energies'] += list(map(float, float_patt.findall(line)))
                        line = file.readline()
                        while coord_patt.match(line):
                            icname = line.split()[0].strip()
                            if icname in data['coords']:
                                data['coords'][icname] += list(map(float, float_patt.findall(line)))
                            else:
                                data['coords'][icname] = list(map(float, float_patt.findall(line)))
                            line = file.readline()
                        if not re.search('^\\s+((\\s*\\d+)+)', line):
                            endScan = True
                        else:
                            line = file.readline()
                elif scan_patt.match(line):
                    line = file.readline()
                    data['coords'] = {icname: [] for icname in line.split()[1:-1]}
                    file.readline()
                    line = file.readline()
                    while not re.search('^\\s-+', line):
                        values = list(map(float, line.split()))
                        data['energies'].append(values[-1])
                        for i, icname in enumerate(data['coords'], start=1):
                            data['coords'][icname].append(values[i])
                        line = file.readline()
                else:
                    line = file.readline()
        return data

    def get_scan_plot(self, coords=None):
        """
        Get a matplotlib plot of the potential energy surface.

        Args:
            coords: internal coordinate name to use as abscissa.
        """
        ax = pretty_plot(12, 8)
        dct = self.read_scan()
        if coords and coords in dct['coords']:
            x = dct['coords'][coords]
            ax.set_xlabel(coords)
        else:
            x = range(len(dct['energies']))
            ax.set_xlabel('points')
        ax.set_ylabel('Energy (eV)')
        e_min = min(dct['energies'])
        y = [(e - e_min) * Ha_to_eV for e in dct['energies']]
        ax.plot(x, y, 'ro--')
        return ax

    def save_scan_plot(self, filename='scan.pdf', img_format='pdf', coords=None):
        """
        Save matplotlib plot of the potential energy surface to a file.

        Args:
            filename: Filename to write to.
            img_format: Image format to use. Defaults to EPS.
            coords: internal coordinate name to use as abcissa.
        """
        plt = self.get_scan_plot(coords)
        plt.savefig(filename, format=img_format)

    def read_excitation_energies(self):
        """
        Read a excitation energies after a TD-DFT calculation.

        Returns:
            A list: A list of tuple for each transition such as
                    [(energie (eV), lambda (nm), oscillatory strength), ... ]
        """
        transitions = []
        with zopen(self.filename, mode='r') as file:
            line = file.readline()
            td = False
            while line != '':
                if re.search('^\\sExcitation energies and oscillator strengths:', line):
                    td = True
                if td and re.search('^\\sExcited State\\s*\\d', line):
                    val = [float(v) for v in float_patt.findall(line)]
                    transitions.append(tuple(val[:3]))
                line = file.readline()
        return transitions

    def get_spectre_plot(self, sigma=0.05, step=0.01):
        """
        Get a matplotlib plot of the UV-visible xas. Transitions are plotted
        as vertical lines and as a sum of normal functions with sigma with. The
        broadening is applied in energy and the xas is plotted as a function
        of the wavelength.

        Args:
            sigma: Full width at half maximum in eV for normal functions.
            step: bin interval in eV

        Returns:
            A dict: {"energies": values, "lambda": values, "xas": values}
                    where values are lists of abscissa (energies, lamba) and
                    the sum of gaussian functions (xas).
            A matplotlib plot.
        """
        ax = pretty_plot(12, 8)
        transitions = self.read_excitation_energies()
        minval = min((val[0] for val in transitions)) - 5.0 * sigma
        maxval = max((val[0] for val in transitions)) + 5.0 * sigma
        npts = int((maxval - minval) / step) + 1
        eneval = np.linspace(minval, maxval, npts)
        lambdaval = [cst.h * cst.c / (val * cst.e) * 1000000000.0 for val in eneval]
        spectre = np.zeros(npts)
        for trans in transitions:
            spectre += trans[2] * norm(eneval, trans[0], sigma)
        spectre /= spectre.max()
        ax.plot(lambdaval, spectre, 'r-', label='spectre')
        data = {'energies': eneval, 'lambda': lambdaval, 'xas': spectre}
        ax.vlines([val[1] for val in transitions], 0.0, [val[2] for val in transitions], color='blue', label='transitions', linewidth=2)
        ax.set_xlabel('$\\lambda$ (nm)')
        ax.set_ylabel('Arbitrary unit')
        ax.legend()
        return (data, ax)

    def save_spectre_plot(self, filename='spectre.pdf', img_format='pdf', sigma=0.05, step=0.01):
        """
        Save matplotlib plot of the spectre to a file.

        Args:
            filename: Filename to write to.
            img_format: Image format to use. Defaults to EPS.
            sigma: Full width at half maximum in eV for normal functions.
            step: bin interval in eV
        """
        _d, plt = self.get_spectre_plot(sigma, step)
        plt.savefig(filename, format=img_format)

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