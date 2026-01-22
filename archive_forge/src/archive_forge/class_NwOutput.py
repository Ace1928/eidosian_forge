from __future__ import annotations
import os
import re
import warnings
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.analysis.excitation import ExcitationSpectrum
from pymatgen.core.structure import Molecule, Structure
from pymatgen.core.units import Energy, FloatWithUnit
class NwOutput:
    """
    A Nwchem output file parser. Very basic for now - supports only dft and
    only parses energies and geometries. Please note that Nwchem typically
    outputs energies in either au or kJ/mol. All energies are converted to
    eV in the parser.
    """

    def __init__(self, filename):
        """
        Args:
            filename: Filename to read.
        """
        self.filename = filename
        with zopen(filename) as file:
            data = file.read()
        chunks = re.split('NWChem Input Module', data)
        if re.search('CITATION', chunks[-1]):
            chunks.pop()
        preamble = chunks.pop(0)
        self.raw = data
        self.job_info = self._parse_preamble(preamble)
        self.data = [self._parse_job(c) for c in chunks]

    def parse_tddft(self):
        """
        Parses TDDFT roots. Adapted from nw_spectrum.py script.

        Returns:
            {
                "singlet": [
                    {
                        "energy": float,
                        "osc_strength: float
                    }
                ],
                "triplet": [
                    {
                        "energy": float
                    }
                ]
            }
        """
        start_tag = 'Convergence criterion met'
        end_tag = 'Excited state energy'
        singlet_tag = 'singlet excited'
        triplet_tag = 'triplet excited'
        state = 'singlet'
        inside = False
        lines = self.raw.split('\n')
        roots = {'singlet': [], 'triplet': []}
        while lines:
            line = lines.pop(0).strip()
            if start_tag in line:
                inside = True
            elif end_tag in line:
                inside = False
            elif singlet_tag in line:
                state = 'singlet'
            elif triplet_tag in line:
                state = 'triplet'
            elif inside and 'Root' in line and ('eV' in line):
                tokens = line.split()
                roots[state].append({'energy': float(tokens[-2])})
            elif inside and 'Dipole Oscillator Strength' in line:
                osc = float(line.split()[-1])
                roots[state][-1]['osc_strength'] = osc
        return roots

    def get_excitation_spectrum(self, width=0.1, npoints=2000):
        """
        Generate an excitation spectra from the singlet roots of TDDFT calculations.

        Args:
            width (float): Width for Gaussian smearing.
            npoints (int): Number of energy points. More points => smoother
                curve.

        Returns:
            ExcitationSpectrum: can be plotted using pymatgen.vis.plotters.SpectrumPlotter.
        """
        roots = self.parse_tddft()
        data = roots['singlet']
        en = np.array([d['energy'] for d in data])
        osc = np.array([d['osc_strength'] for d in data])
        epad = 20.0 * width
        emin = en[0] - epad
        emax = en[-1] + epad
        de = (emax - emin) / npoints
        if width < 2 * de:
            width = 2 * de
        energies = [emin + ie * de for ie in range(npoints)]
        cutoff = 20.0 * width
        gamma = 0.5 * width
        gamma_sqrd = gamma * gamma
        de = (energies[-1] - energies[0]) / (len(energies) - 1)
        prefac = gamma / np.pi * de
        x = []
        y = []
        for energy in energies:
            xx0 = energy - en
            stot = osc / (xx0 * xx0 + gamma_sqrd)
            t = np.sum(stot[np.abs(xx0) <= cutoff])
            x.append(energy)
            y.append(t * prefac)
        return ExcitationSpectrum(x, y)

    @staticmethod
    def _parse_preamble(preamble):
        info = {}
        for line in preamble.split('\n'):
            tokens = line.split('=')
            if len(tokens) > 1:
                info[tokens[0].strip()] = tokens[-1].strip()
        return info

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _parse_job(output):
        energy_patt = re.compile('Total \\w+ energy\\s+=\\s+([.\\-\\d]+)')
        energy_gas_patt = re.compile('gas phase energy\\s+=\\s+([.\\-\\d]+)')
        energy_sol_patt = re.compile('sol phase energy\\s+=\\s+([.\\-\\d]+)')
        coord_patt = re.compile('\\d+\\s+(\\w+)\\s+[.\\-\\d]+\\s+([.\\-\\d]+)\\s+([.\\-\\d]+)\\s+([.\\-\\d]+)')
        lat_vector_patt = re.compile('a[123]=<\\s+([.\\-\\d]+)\\s+([.\\-\\d]+)\\s+([.\\-\\d]+)\\s+>')
        corrections_patt = re.compile('([\\w\\-]+ correction to \\w+)\\s+=\\s+([.\\-\\d]+)')
        preamble_patt = re.compile('(No. of atoms|No. of electrons|SCF calculation type|Charge|Spin multiplicity)\\s*:\\s*(\\S+)')
        force_patt = re.compile('\\s+(\\d+)\\s+(\\w+)' + 6 * '\\s+([0-9\\.\\-]+)')
        time_patt = re.compile('\\s+ Task \\s+ times \\s+ cpu: \\s+   ([.\\d]+)s .+ ', re.VERBOSE)
        error_defs = {'calculations not reaching convergence': 'Bad convergence', 'Calculation failed to converge': 'Bad convergence', 'geom_binvr: #indep variables incorrect': 'autoz error', 'dft optimize failed': 'Geometry optimization failed'}

        def fort2py(x):
            return x.replace('D', 'e')

        def isfloatstring(in_str):
            return in_str.find('.') == -1
        parse_hess = False
        parse_proj_hess = False
        hessian = projected_hessian = None
        parse_force = False
        all_forces = []
        forces = []
        data = {}
        energies = []
        frequencies = normal_frequencies = None
        corrections = {}
        molecules = []
        structures = []
        species = []
        coords = []
        lattice = []
        errors = []
        basis_set = {}
        bset_header = []
        parse_geom = False
        parse_freq = False
        parse_bset = False
        parse_projected_freq = False
        job_type = ''
        parse_time = False
        time = 0
        for line in output.split('\n'):
            for e, v in error_defs.items():
                if line.find(e) != -1:
                    errors.append(v)
            if parse_time:
                m = time_patt.search(line)
                if m:
                    time = m.group(1)
                    parse_time = False
            if parse_geom:
                if line.strip() == 'Atomic Mass':
                    if lattice:
                        structures.append(Structure(lattice, species, coords, coords_are_cartesian=True))
                    else:
                        molecules.append(Molecule(species, coords))
                    species = []
                    coords = []
                    lattice = []
                    parse_geom = False
                else:
                    m = coord_patt.search(line)
                    if m:
                        species.append(m.group(1).capitalize())
                        coords.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
                    m = lat_vector_patt.search(line)
                    if m:
                        lattice.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            if parse_force:
                m = force_patt.search(line)
                if m:
                    forces.extend(map(float, m.groups()[5:]))
                elif len(forces) > 0:
                    all_forces.append(forces)
                    forces = []
                    parse_force = False
            elif parse_freq:
                if len(line.strip()) == 0:
                    if len(normal_frequencies[-1][1]) == 0:
                        continue
                    parse_freq = False
                else:
                    vibs = [float(vib) for vib in line.strip().split()[1:]]
                    n_vibs = len(vibs)
                    for mode, dis in zip(normal_frequencies[-n_vibs:], vibs):
                        mode[1].append(dis)
            elif parse_projected_freq:
                if len(line.strip()) == 0:
                    if len(frequencies[-1][1]) == 0:
                        continue
                    parse_projected_freq = False
                else:
                    vibs = [float(vib) for vib in line.strip().split()[1:]]
                    n_vibs = len(vibs)
                    for mode, dis in zip(frequencies[-n_vibs:], vibs):
                        mode[1].append(dis)
            elif parse_bset:
                if line.strip() == '':
                    parse_bset = False
                else:
                    tokens = line.split()
                    if tokens[0] != 'Tag' and (not re.match('-+', tokens[0])):
                        basis_set[tokens[0]] = dict(zip(bset_header[1:], tokens[1:]))
                    elif tokens[0] == 'Tag':
                        bset_header = tokens
                        bset_header.pop(4)
                        bset_header = [h.lower() for h in bset_header]
            elif parse_hess:
                if line.strip() == '':
                    continue
                if len(hessian) > 0 and line.find('----------') != -1:
                    parse_hess = False
                    continue
                tokens = line.strip().split()
                if len(tokens) > 1:
                    try:
                        row = int(tokens[0])
                    except Exception:
                        continue
                    if isfloatstring(tokens[1]):
                        continue
                    vals = [float(fort2py(x)) for x in tokens[1:]]
                    if len(hessian) < row:
                        hessian.append(vals)
                    else:
                        hessian[row - 1].extend(vals)
            elif parse_proj_hess:
                if line.strip() == '':
                    continue
                nat3 = len(hessian)
                tokens = line.strip().split()
                if len(tokens) > 1:
                    try:
                        row = int(tokens[0])
                    except Exception:
                        continue
                    if isfloatstring(tokens[1]):
                        continue
                    vals = [float(fort2py(x)) for x in tokens[1:]]
                    if len(projected_hessian) < row:
                        projected_hessian.append(vals)
                    else:
                        projected_hessian[row - 1].extend(vals)
                    if len(projected_hessian[-1]) == nat3:
                        parse_proj_hess = False
            else:
                m = energy_patt.search(line)
                if m:
                    energies.append(Energy(m.group(1), 'Ha').to('eV'))
                    parse_time = True
                    continue
                m = energy_gas_patt.search(line)
                if m:
                    cosmo_scf_energy = energies[-1]
                    energies[-1] = {}
                    energies[-1]['cosmo scf'] = cosmo_scf_energy
                    energies[-1].update({'gas phase': Energy(m.group(1), 'Ha').to('eV')})
                m = energy_sol_patt.search(line)
                if m:
                    energies[-1].update({'sol phase': Energy(m.group(1), 'Ha').to('eV')})
                m = preamble_patt.search(line)
                if m:
                    try:
                        val = int(m.group(2))
                    except ValueError:
                        val = m.group(2)
                    k = m.group(1).replace('No. of ', 'n').replace(' ', '_')
                    data[k.lower()] = val
                elif line.find('Geometry "geometry"') != -1:
                    parse_geom = True
                elif line.find('Summary of "ao basis"') != -1:
                    parse_bset = True
                elif line.find('P.Frequency') != -1:
                    parse_projected_freq = True
                    if frequencies is None:
                        frequencies = []
                    tokens = line.strip().split()[1:]
                    frequencies.extend([(float(freq), []) for freq in tokens])
                elif line.find('Frequency') != -1:
                    tokens = line.strip().split()
                    if len(tokens) > 1 and tokens[0] == 'Frequency':
                        parse_freq = True
                        if normal_frequencies is None:
                            normal_frequencies = []
                        normal_frequencies.extend([(float(freq), []) for freq in line.strip().split()[1:]])
                elif line.find('MASS-WEIGHTED NUCLEAR HESSIAN') != -1:
                    parse_hess = True
                    if not hessian:
                        hessian = []
                elif line.find('MASS-WEIGHTED PROJECTED HESSIAN') != -1:
                    parse_proj_hess = True
                    if not projected_hessian:
                        projected_hessian = []
                elif line.find('atom               coordinates                        gradient') != -1:
                    parse_force = True
                elif job_type == '' and line.strip().startswith('NWChem'):
                    job_type = line.strip()
                    if job_type == 'NWChem DFT Module' and 'COSMO solvation results' in output:
                        job_type += ' COSMO'
                else:
                    m = corrections_patt.search(line)
                    if m:
                        corrections[m.group(1)] = FloatWithUnit(m.group(2), 'kJ mol^-1').to('eV atom^-1')
        if frequencies:
            for _freq, mode in frequencies:
                mode[:] = zip(*[iter(mode)] * 3)
        if normal_frequencies:
            for _freq, mode in normal_frequencies:
                mode[:] = zip(*[iter(mode)] * 3)
        if hessian:
            len_hess = len(hessian)
            for ii in range(len_hess):
                for jj in range(ii + 1, len_hess):
                    hessian[ii].append(hessian[jj][ii])
        if projected_hessian:
            len_hess = len(projected_hessian)
            for ii in range(len_hess):
                for jj in range(ii + 1, len_hess):
                    projected_hessian[ii].append(projected_hessian[jj][ii])
        data.update({'job_type': job_type, 'energies': energies, 'corrections': corrections, 'molecules': molecules, 'structures': structures, 'basis_set': basis_set, 'errors': errors, 'has_error': len(errors) > 0, 'frequencies': frequencies, 'normal_frequencies': normal_frequencies, 'hessian': hessian, 'projected_hessian': projected_hessian, 'forces': all_forces, 'task_time': time})
        return data