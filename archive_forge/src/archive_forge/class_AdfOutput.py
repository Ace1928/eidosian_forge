from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
from monty.io import reverse_readline
from monty.itertools import chunks
from monty.json import MSONable
from monty.serialization import zopen
from pymatgen.core.structure import Molecule
class AdfOutput:
    """
    A basic ADF output file parser.

    Attributes:
        is_failed (bool): Whether the ADF job is failed.
        is_internal_crash (bool): Whether the job crashed.
            Please read 'TAPE13' of the ADF manual for more detail.
        error (str): The error description.
        run_type (str): The RunType of this ADF job. Possible options are:
            'SinglePoint', 'GeometryOptimization', 'AnalyticalFreq' and 'NUmericalFreq'.
        final_energy (float): The final molecule energy (a.u).
        final_structure (GMolecule): The final structure of the molecule.
        energies (Sized): The energy of each cycle.
        structures (Sized): The structure of each cycle If geometry optimization is performed.
        frequencies (array_like): The frequencies of the molecule.
        normal_modes (array_like): The normal modes of the molecule.
        freq_type (syr): Either 'Analytical' or 'Numerical'.
    """

    def __init__(self, filename):
        """
        Initialization method.

        Args:
            filename (str): The ADF output file to parse.
        """
        self.filename = filename
        self._parse()

    def _parse(self):
        """
        Parse the ADF outputs. There are two files: one is 'logfile', the other
        is the ADF output file. The final energy and structures are parsed from
        the 'logfile'. Frequencies and normal modes are parsed from the ADF
        output file.
        """
        workdir = os.path.dirname(self.filename)
        logfile = f'{workdir}/logfile'
        for ext in ('', '.gz', '.bz2'):
            if os.path.isfile(f'{logfile}{ext}'):
                logfile = f'{logfile}{ext}'
                break
        else:
            raise FileNotFoundError('The ADF logfile can not be accessed!')
        self.is_failed = False
        self.error = self.final_energy = self.final_structure = None
        self.energies = []
        self.structures = []
        self.frequencies = []
        self.normal_modes = self.freq_type = self.run_type = None
        self.is_internal_crash = False
        self._parse_logfile(logfile)
        if not self.is_failed and self.run_type != 'SinglePoint':
            self._parse_adf_output()

    @staticmethod
    def _sites_to_mol(sites):
        """
        Return a ``Molecule`` object given a list of sites.

        Args:
            sites : A list of sites.

        Returns:
            mol (Molecule): A ``Molecule`` object.
        """
        return Molecule([site[0] for site in sites], [site[1] for site in sites])

    def _parse_logfile(self, logfile):
        """Parse the formatted logfile."""
        cycle_patt = re.compile('Coordinates\\sin\\sGeometry\\sCycle\\s(\\d+)')
        coord_patt = re.compile('\\s+([0-9]+)\\.([A-Za-z]+)' + 3 * '\\s+([-\\.0-9]+)')
        energy_patt = re.compile('<.*>\\s<.*>\\s+current\\senergy\\s+([-\\.0-9]+)\\sHartree')
        final_energy_patt = re.compile('<.*>\\s<.*>\\s+Bond\\sEnergy\\s+([-\\.0-9]+)\\sa\\.u\\.')
        error_patt = re.compile('<.*>\\s<.*>\\s+ERROR\\sDETECTED:\\s(.*)')
        run_type_patt = re.compile('<.*>\\s<.*>\\s+RunType\\s+:\\s(.*)')
        end_patt = re.compile('<.*>\\s<.*>\\s+END')
        parse_cycle = False
        sites = []
        last_cycle = -1
        parse_final = False
        with zopen(logfile, mode='rt') as file:
            for line in reverse_readline(file):
                if line == '':
                    continue
                if end_patt.search(line) is None:
                    self.is_internal_crash = True
                    self.error = 'Internal crash. TAPE13 is generated!'
                    self.is_failed = True
                    return
                break
        with open(logfile) as file:
            for line in file:
                m = error_patt.search(line)
                if m:
                    self.is_failed = True
                    self.error = m.group(1)
                    break
                if self.run_type is None:
                    m = run_type_patt.search(line)
                    if m:
                        if m.group(1) == 'FREQUENCIES':
                            self.freq_type = 'Numerical'
                            self.run_type = 'NumericalFreq'
                        elif m.group(1) == 'GEOMETRY OPTIMIZATION':
                            self.run_type = 'GeometryOptimization'
                        elif m.group(1) == 'CREATE':
                            self.run_type = None
                        elif m.group(1) == 'SINGLE POINT':
                            self.run_type = 'SinglePoint'
                        else:
                            raise AdfOutputError('Undefined Runtype!')
                elif self.run_type == 'SinglePoint':
                    m = coord_patt.search(line)
                    if m:
                        sites.append([m.groups()[0], list(map(float, m.groups()[2:]))])
                    else:
                        m = final_energy_patt.search(line)
                        if m:
                            self.final_energy = float(m.group(1))
                            self.final_structure = self._sites_to_mol(sites)
                elif self.run_type == 'GeometryOptimization':
                    m = cycle_patt.search(line)
                    if m:
                        cycle = int(m.group(1))
                        if cycle <= 0:
                            raise AdfOutputError(f'Wrong cycle={cycle!r}')
                        if cycle > last_cycle:
                            parse_cycle = True
                            last_cycle = cycle
                        else:
                            parse_final = True
                    elif parse_cycle:
                        m = coord_patt.search(line)
                        if m:
                            sites.append([m.groups()[1], list(map(float, m.groups()[2:]))])
                        else:
                            m = energy_patt.search(line)
                            if m:
                                self.energies.append(float(m.group(1)))
                                mol = self._sites_to_mol(sites)
                                self.structures.append(mol)
                                parse_cycle = False
                                sites = []
                    elif parse_final:
                        m = final_energy_patt.search(line)
                        if m:
                            self.final_energy = float(m.group(1))
                elif self.run_type == 'NumericalFreq':
                    break
        if not self.is_failed:
            if self.run_type == 'GeometryOptimization':
                if len(self.structures) > 0:
                    self.final_structure = self.structures[-1]
                if self.final_energy is None:
                    raise AdfOutputError('The final energy can not be read!')
            elif self.run_type == 'SinglePoint':
                if self.final_structure is None:
                    raise AdfOutputError('The final structure is missing!')
                if self.final_energy is None:
                    raise AdfOutputError('The final energy can not be read!')

    def _parse_adf_output(self):
        """Parse the standard ADF output file."""
        numerical_freq_patt = re.compile('\\s+\\*\\s+F\\sR\\sE\\sQ\\sU\\sE\\sN\\sC\\sI\\sE\\sS\\s+\\*')
        analytic_freq_patt = re.compile('\\s+\\*\\s+F\\sR\\sE\\sQ\\sU\\sE\\sN\\sC\\sY\\s+A\\sN\\sA\\sL\\sY\\sS\\sI\\sS\\s+\\*')
        freq_on_patt = re.compile('Vibrations\\sand\\sNormal\\sModes\\s+\\*+.*\\*+')
        freq_off_patt = re.compile('List\\sof\\sAll\\sFrequencies:')
        mode_patt = re.compile('\\s+(\\d+)\\.([A-Za-z]+)\\s+(.*)')
        coord_patt = re.compile('\\s+(\\d+)\\s+([A-Za-z]+)' + 6 * '\\s+([0-9\\.-]+)')
        coord_on_patt = re.compile('\\s+\\*\\s+R\\sU\\sN\\s+T\\sY\\sP\\sE\\s:\\sFREQUENCIES\\s+\\*')
        parse_freq = False
        parse_mode = False
        n_next = 0
        n_strike = 0
        sites = []
        self.frequencies = []
        self.normal_modes = []
        if self.final_structure is None:
            find_structure = True
            parse_coord = False
            n_atoms = 0
        else:
            find_structure = False
            parse_coord = False
            n_atoms = len(self.final_structure)
        with open(self.filename) as file:
            for line in file:
                if self.run_type == 'NumericalFreq' and find_structure:
                    if not parse_coord:
                        m = coord_on_patt.search(line)
                        if m:
                            parse_coord = True
                    else:
                        m = coord_patt.search(line)
                        if m:
                            sites.append([m.group(2), list(map(float, m.groups()[2:5]))])
                            n_strike += 1
                        elif n_strike > 0:
                            find_structure = False
                            self.final_structure = self._sites_to_mol(sites)
                            n_atoms = len(self.final_structure)
                elif self.freq_type is None:
                    if numerical_freq_patt.search(line):
                        self.freq_type = 'Numerical'
                    elif analytic_freq_patt.search(line):
                        self.freq_type = 'Analytical'
                        self.run_type = 'AnalyticalFreq'
                elif freq_on_patt.search(line):
                    parse_freq = True
                elif parse_freq:
                    if freq_off_patt.search(line):
                        break
                    el = line.strip().split()
                    if 1 <= len(el) <= 3 and line.find('.') != -1:
                        n_next = len(el)
                        parse_mode = True
                        parse_freq = False
                        self.frequencies.extend(map(float, el))
                        for _ in range(n_next):
                            self.normal_modes.append([])
                elif parse_mode:
                    m = mode_patt.search(line)
                    if m:
                        v = list(chunks(map(float, m.group(3).split()), 3))
                        if len(v) != n_next:
                            raise AdfOutputError('Odd Error!')
                        for i, k in enumerate(range(-n_next, 0)):
                            self.normal_modes[k].extend(v[i])
                        if int(m.group(1)) == n_atoms:
                            parse_freq = True
                            parse_mode = False
        if isinstance(self.final_structure, list):
            self.final_structure = self._sites_to_mol(self.final_structure)
        if self.freq_type is not None:
            if len(self.frequencies) != len(self.normal_modes):
                raise AdfOutputError('The number of normal modes is wrong!')
            if len(self.normal_modes[0]) != n_atoms * 3:
                raise AdfOutputError('The dimensions of the modes are wrong!')