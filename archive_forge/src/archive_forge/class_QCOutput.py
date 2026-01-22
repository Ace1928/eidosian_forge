from __future__ import annotations
import copy
import logging
import math
import os
import re
import struct
import warnings
from typing import TYPE_CHECKING, Any
import networkx as nx
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable, jsanitize
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.core import Molecule
from pymatgen.io.qchem.utils import (
class QCOutput(MSONable):
    """Class to parse QChem output files."""

    def __init__(self, filename: str):
        """
        Args:
            filename (str): Filename to parse.
        """
        self.filename = filename
        self.data: dict[str, Any] = {}
        self.data['errors'] = []
        self.data['warnings'] = {}
        self.text = ''
        with zopen(filename, mode='rt', encoding='ISO-8859-1') as file:
            self.text = file.read()
        self.data['multiple_outputs'] = read_pattern(self.text, {'key': 'Job\\s+\\d+\\s+of\\s+(\\d+)\\s+'}, terminate_on_match=True).get('key')
        if self.data.get('multiple_outputs') is not None and self.data.get('multiple_outputs') != [['1']]:
            raise ValueError(f'ERROR: multiple calculation outputs found in file {filename}. Please instead call QCOutput.mulitple_outputs_from_file(QCOutput, {filename!r})')
        if read_pattern(self.text, {'key': 'A Quantum Leap Into The Future Of Chemistry\\s+Q-Chem 4'}, terminate_on_match=True).get('key') == [[]]:
            self.data['version'] = '4'
        elif read_pattern(self.text, {'key': 'A Quantum Leap Into The Future Of Chemistry\\s+Q-Chem 5'}, terminate_on_match=True).get('key') == [[]]:
            self.data['version'] = '5'
        elif read_pattern(self.text, {'key': 'A Quantum Leap Into The Future Of Chemistry\\s+Q-Chem 6'}, terminate_on_match=True).get('key') == [[]]:
            self.data['version'] = '6'
        else:
            self.data['version'] = 'unknown'
        self._read_charge_and_multiplicity()
        if read_pattern(self.text, {'key': 'Nuclear Repulsion Energy'}, terminate_on_match=True).get('key') == [[]]:
            self._read_species_and_inital_geometry()
        self.data['completion'] = read_pattern(self.text, {'key': 'Thank you very much for using Q-Chem.\\s+Have a nice day.'}, terminate_on_match=True).get('key')
        if self.data.get('completion', []):
            temp_timings = read_pattern(self.text, {'key': 'Total job time\\:\\s*([\\d\\-\\.]+)s\\(wall\\)\\,\\s*([\\d\\-\\.]+)s\\(cpu\\)'}).get('key')
            if temp_timings is not None:
                self.data['walltime'] = float(temp_timings[0][0])
                self.data['cputime'] = float(temp_timings[0][1])
            else:
                self.data['walltime'] = self.data['cputime'] = None
        self.data['unrestricted'] = read_pattern(self.text, {'key': 'A(?:n)*\\sunrestricted[\\s\\w\\-]+SCF\\scalculation\\swill\\sbe'}, terminate_on_match=True).get('key')
        if not self.data['unrestricted']:
            self.data['unrestricted'] = read_pattern(self.text, {'key': 'unrestricted = true'}, terminate_on_match=True).get('key')
        if self.data['unrestricted'] is None and self.data['multiplicity'] != 1:
            self.data['unrestricted'] = [[]]
        scf_final_print = read_pattern(self.text, {'key': 'scf_final_print\\s*=\\s*(\\d+)'}, terminate_on_match=True).get('key')
        if scf_final_print is not None:
            self.data['scf_final_print'] = int(scf_final_print[0][0])
        else:
            self.data['scf_final_print'] = 0
        self.data['using_GEN_SCFMAN'] = read_pattern(self.text, {'key': '\\s+GEN_SCFMAN: A general SCF calculation manager'}, terminate_on_match=True).get('key')
        if not self.data['using_GEN_SCFMAN']:
            self.data['using_GEN_SCFMAN'] = read_pattern(self.text, {'key': '\\s+General SCF calculation program by'}, terminate_on_match=True).get('key')
        if read_pattern(self.text, {'key': 'SCF failed to converge'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['SCF_failed_to_converge']
        self._read_SCF()
        self._read_charges_and_dipoles()
        self._detect_general_warnings()
        self.data['mem_total'] = None
        if read_pattern(self.text, {'key': 'mem_total\\s*='}, terminate_on_match=True).get('key') == [[]]:
            temp_mem_total = read_pattern(self.text, {'key': 'mem_total\\s*=\\s*(\\d+)'}, terminate_on_match=True).get('key')
            self.data['mem_total'] = int(temp_mem_total[0][0])
        if read_pattern(self.text, {'key': 'Generalized Kohn-Sham gap'}, terminate_on_match=True).get('key') == [[]]:
            gap_info = {}
            if read_pattern(self.text, {'key': 'Alpha HOMO Eigenvalue'}, terminate_on_match=True).get('key') == [[]]:
                temp_alpha_HOMO = read_pattern(self.text, {'key': 'Alpha HOMO Eigenvalue\\s*=\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
                gap_info['alpha_HOMO'] = float(temp_alpha_HOMO[0][0])
                temp_beta_HOMO = read_pattern(self.text, {'key': 'Beta  HOMO Eigenvalue\\s*=\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
                gap_info['beta_HOMO'] = float(temp_beta_HOMO[0][0])
                temp_alpha_LUMO = read_pattern(self.text, {'key': 'Alpha LUMO Eigenvalue\\s*=\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
                gap_info['alpha_LUMO'] = float(temp_alpha_LUMO[0][0])
                temp_beta_LUMO = read_pattern(self.text, {'key': 'Beta  LUMO Eigenvalue\\s*=\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
                gap_info['beta_LUMO'] = float(temp_beta_LUMO[0][0])
                temp_alpha_gap = read_pattern(self.text, {'key': 'HOMO-Alpha LUMO gap\\s*=\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
                gap_info['alpha_gap'] = float(temp_alpha_gap[0][0])
                temp_beta_gap = read_pattern(self.text, {'key': 'HOMO-Beta LUMO gap\\s*=\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
                gap_info['beta_gap'] = float(temp_beta_gap[0][0])
            temp_HOMO = read_pattern(self.text, {'key': '    HOMO Eigenvalue\\s*=\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
            gap_info['HOMO'] = float(temp_HOMO[0][0])
            temp_LUMO = read_pattern(self.text, {'key': '    LUMO Eigenvalue\\s*=\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
            gap_info['LUMO'] = float(temp_LUMO[0][0])
            temp_KSgap = read_pattern(self.text, {'key': 'KS gap\\s*=\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
            gap_info['KSgap'] = float(temp_KSgap[0][0])
            self.data['gap_info'] = gap_info
        else:
            self.data['gap_info'] = None
        self.data['solvent_method'] = self.data['solvent_data'] = None
        if read_pattern(self.text, {'key': 'solvent_method\\s*=?\\s*pcm'}, terminate_on_match=True).get('key') == [[]]:
            self.data['solvent_method'] = 'PCM'
        if read_pattern(self.text, {'key': 'solvent_method\\s*=?\\s*smd'}, terminate_on_match=True).get('key') == [[]]:
            self.data['solvent_method'] = 'SMD'
        if read_pattern(self.text, {'key': 'solvent_method\\s*=?\\s*isosvp'}, terminate_on_match=True).get('key') == [[]]:
            self.data['solvent_method'] = 'ISOSVP'
        pcm_keys = ['PCM_dielectric', 'g_electrostatic', 'g_cavitation', 'g_dispersion', 'g_repulsion', 'total_contribution_pcm', 'solute_internal_energy']
        smd_keys = ['SMD_solvent', 'smd0', 'smd3', 'smd4', 'smd6', 'smd9']
        isosvp_keys = ['isosvp_dielectric', 'final_soln_phase_e', 'solute_internal_e', 'total_solvation_free_e', 'change_solute_internal_e', 'reaction_field_free_e']
        cmirs_keys = ['CMIRS_enabled', 'dispersion_e', 'exchange_e', 'min_neg_field_e', 'max_pos_field_e']
        if self.data['solvent_method'] is not None:
            self.data['solvent_data'] = {}
            for key in pcm_keys + smd_keys:
                self.data['solvent_data'][key] = None
            self.data['solvent_data']['isosvp'] = {}
            for key in isosvp_keys:
                self.data['solvent_data']['isosvp'][key] = None
            self.data['solvent_data']['cmirs'] = {}
            for key in cmirs_keys:
                self.data['solvent_data']['cmirs'][key] = None
        if self.data['solvent_method'] == 'PCM':
            temp_dielectric = read_pattern(self.text, {'key': 'dielectric\\s*([\\d\\-\\.]+)'}, terminate_on_match=True).get('key')
            self.data['solvent_data']['PCM_dielectric'] = float(temp_dielectric[0][0])
            self._read_pcm_information()
        elif self.data['solvent_method'] == 'SMD':
            if read_pattern(self.text, {'key': 'Unrecognized solvent'}, terminate_on_match=True).get('key') == [[]]:
                if not self.data.get('completion', []):
                    self.data['errors'] += ['unrecognized_solvent']
                else:
                    self.data['warnings']['unrecognized_solvent'] = True
            temp_solvent = read_pattern(self.text, {'key': '\\s[Ss]olvent:? ([a-zA-Z]+)'}).get('key')
            for val in temp_solvent:
                if val[0] != temp_solvent[0][0]:
                    if val[0] != 'for':
                        self.data['warnings']['SMD_two_solvents'] = f'{temp_solvent[0][0]} and {val[0]}'
                    elif 'unrecognized_solvent' not in self.data['errors'] and 'unrecognized_solvent' not in self.data['warnings']:
                        self.data['warnings']['questionable_SMD_parsing'] = True
            self.data['solvent_data']['SMD_solvent'] = temp_solvent[0][0]
            self._read_smd_information()
        elif self.data['solvent_method'] == 'ISOSVP':
            self.data['solvent_data']['cmirs']['CMIRS_enabled'] = False
            self._read_isosvp_information()
            if read_pattern(self.text, {'cmirs': 'DEFESR calculation with single-center isodensity surface'}, terminate_on_match=True).get('cmirs') == [[]]:
                self._read_cmirs_information()
                self.data['solvent_data']['cmirs']['CMIRS_enabled'] = True
        temp_final_energy = read_pattern(self.text, {'key': 'Final\\senergy\\sis\\s+([\\d\\-\\.]+)'}).get('key')
        if temp_final_energy is None:
            self.data['final_energy'] = None
        else:
            self.data['final_energy'] = float(temp_final_energy[0][0])
        if self.data['final_energy'] is None:
            temp_dict = read_pattern(self.text, {'final_energy': '\\s*Total\\s+energy in the final basis set\\s+=\\s*([\\d\\-\\.]+)'}) or read_pattern(self.text, {'final_energy': '\\s+Total energy\\s+=\\s+([\\d\\-\\.]+)'})
            if (e_final_match := temp_dict.get('final_energy')):
                self.data['final_energy'] = float(e_final_match[-1][0])
        self.data['using_dft_d3'] = read_pattern(self.text, {'key': 'dft_d\\s*= d3'}, terminate_on_match=True).get('key')
        if self.data.get('using_dft_d3', []):
            temp_d3 = read_pattern(self.text, {'key': '\\-D3 energy without 3body term =\\s*([\\d\\.\\-]+) hartrees'}).get('key')
            real_d3 = np.zeros(len(temp_d3))
            if temp_d3 is None:
                self.data['dft_d3'] = None
            elif len(temp_d3) == 1:
                self.data['dft_d3'] = float(temp_d3[0][0])
            else:
                for ii, entry in enumerate(temp_d3):
                    real_d3[ii] = float(entry[0])
                self.data['dft_d3'] = real_d3
        if self.data.get('unrestricted', []):
            correct_s2 = 0.5 * (self.data['multiplicity'] - 1) * (0.5 * (self.data['multiplicity'] - 1) + 1)
            temp_S2 = read_pattern(self.text, {'key': '<S\\^2>\\s=\\s+([\\d\\-\\.]+)'}).get('key')
            if temp_S2 is None:
                self.data['S2'] = None
            elif len(temp_S2) == 1:
                self.data['S2'] = float(temp_S2[0][0])
                if abs(correct_s2 - self.data['S2']) > 0.01:
                    self.data['warnings']['spin_contamination'] = abs(correct_s2 - self.data['S2'])
            else:
                real_S2 = np.zeros(len(temp_S2))
                have_spin_contamination = False
                for ii, entry in enumerate(temp_S2):
                    real_S2[ii] = float(entry[0])
                    if abs(correct_s2 - real_S2[ii]) > 0.01:
                        have_spin_contamination = True
                self.data['S2'] = real_S2
                if have_spin_contamination:
                    spin_contamination = np.zeros(len(self.data['S2']))
                    for ii, entry in enumerate(self.data['S2']):
                        spin_contamination[ii] = abs(correct_s2 - entry)
                    self.data['warnings']['spin_contamination'] = spin_contamination
        self.data['cdft'] = read_pattern(self.text, {'key': 'CDFT Becke Populations'}).get('key')
        if self.data.get('cdft', []):
            self._read_cdft()
        self.data['cdft_direct_coupling'] = read_pattern(self.text, {'key': 'Start with Direct-Coupling Calculation'}).get('key')
        if self.data.get('cdft_direct_coupling', []):
            temp_dict = read_pattern(self.text, {'Hif': '\\s*DC Matrix Element\\s+Hif =\\s+([\\-\\.0-9]+)', 'Sif': '\\s*DC Matrix Element\\s+Sif =\\s+([\\-\\.0-9]+)', 'Hii': '\\s*DC Matrix Element\\s+Hii =\\s+([\\-\\.0-9]+)', 'Sii': '\\s*DC Matrix Element\\s+Sii =\\s+([\\-\\.0-9]+)', 'Hff': '\\s*DC Matrix Element\\s+Hff =\\s+([\\-\\.0-9]+)', 'Sff': '\\s*DC Matrix Element\\s+Sff =\\s+([\\-\\.0-9]+)', 'coupling': '\\s*Effective Coupling \\(in eV\\) =\\s+([\\-\\.0-9]+)'})
            if len(temp_dict.get('Hif', [])) == 0:
                self.data['direct_coupling_Hif_Hartree'] = None
            else:
                self.data['direct_coupling_Hif_Hartree'] = float(temp_dict['Hif'][0][0])
            if len(temp_dict.get('Sif', [])) == 0:
                self.data['direct_coupling_Sif_Hartree'] = None
            else:
                self.data['direct_coupling_Sif_Hartree'] = float(temp_dict['Sif'][0][0])
            if len(temp_dict.get('Hii', [])) == 0:
                self.data['direct_coupling_Hii_Hartree'] = None
            else:
                self.data['direct_coupling_Hii_Hartree'] = float(temp_dict['Hii'][0][0])
            if len(temp_dict.get('Sii', [])) == 0:
                self.data['direct_coupling_Sii_Hartree'] = None
            else:
                self.data['direct_coupling_Sii_Hartree'] = float(temp_dict['Sii'][0][0])
            if len(temp_dict.get('Hff', [])) == 0:
                self.data['direct_coupling_Hff_Hartree'] = None
            else:
                self.data['direct_coupling_Hff_Hartree'] = float(temp_dict['Hff'][0][0])
            if len(temp_dict.get('Sff', [])) == 0:
                self.data['direct_coupling_Sff_Hartree'] = None
            else:
                self.data['direct_coupling_Sff_Hartree'] = float(temp_dict['Sff'][0][0])
            if len(temp_dict.get('coupling', [])) == 0:
                self.data['direct_coupling_eV'] = None
            else:
                self.data['direct_coupling_eV'] = float(temp_dict['coupling'][0][0])
        self.data['almo_msdft'] = read_pattern(self.text, {'key': 'ALMO\\(MSDFT2?\\) method for electronic coupling'}).get('key')
        if self.data.get('almo_msdft', []):
            self._read_almo_msdft()
        self.data['pod'] = read_pattern(self.text, {'key': 'POD2? based on the RSCF Fock matrix'}).get('key')
        if self.data.get('pod', []):
            coupling = read_pattern(self.text, {'coupling': 'The D\\([0-9]+\\) \\- A\\([0-9]+\\) coupling:\\s+(?:[\\.\\-0-9]+ \\()?([\\-\\.0-9]+) meV\\)?'}).get('coupling')
            if coupling is None or len(coupling) == 0:
                self.data['pod_coupling_eV'] = None
            else:
                self.data['pod_coupling_eV'] = float(coupling[0][0]) / 1000
        self.data['fodft'] = read_pattern(self.text, {'key': 'FODFT\\(2n(?:[\\-\\+]1)?\\)\\@D(?:\\^[\\-\\+])?A(?:\\^\\-)? for [EH]T'}).get('key')
        if self.data.get('fodft', []):
            temp_dict = read_pattern(self.text, {'had': 'H_ad = (?:[\\-\\.0-9]+) \\(([\\-\\.0-9]+) meV\\)', 'hda': 'H_da = (?:[\\-\\.0-9]+) \\(([\\-\\.0-9]+) meV\\)', 'coupling': 'The (?:averaged )?electronic coupling: (?:[\\-\\.0-9]+) \\(([\\-\\.0-9]+) meV\\)'})
            if temp_dict.get('had') is None or len(temp_dict.get('had', [])) == 0:
                self.data['fodft_had_eV'] = None
            else:
                self.data['fodft_had_eV'] = float(temp_dict['had'][0][0]) / 1000
            if temp_dict.get('hda') is None or len(temp_dict.get('hda', [])) == 0:
                self.data['fodft_hda_eV'] = None
            else:
                self.data['fodft_hda_eV'] = float(temp_dict['hda'][0][0]) / 1000
            if temp_dict.get('coupling') is None or len(temp_dict.get('coupling', [])) == 0:
                self.data['fodft_coupling_eV'] = None
            else:
                self.data['fodft_coupling_eV'] = float(temp_dict['coupling'][0][0]) / 1000
        self.data['coupled_cluster'] = read_pattern(self.text, {'key': 'CCMAN2: suite of methods based on coupled cluster'}).get('key')
        if self.data.get('coupled_cluster', []):
            temp_dict = read_pattern(self.text, {'SCF': '\\s+SCF energy\\s+=\\s+([\\d\\-\\.]+)', 'MP2': '\\s+MP2 energy\\s+=\\s+([\\d\\-\\.]+)', 'CCSD_correlation': '\\s+CCSD correlation energy\\s+=\\s+([\\d\\-\\.]+)', 'CCSD': '\\s+CCSD total energy\\s+=\\s+([\\d\\-\\.]+)', 'CCSD(T)_correlation': '\\s+CCSD\\(T\\) correlation energy\\s+=\\s+([\\d\\-\\.]+)', 'CCSD(T)': '\\s+CCSD\\(T\\) total energy\\s+=\\s+([\\d\\-\\.]+)'})
            if temp_dict.get('SCF') is None:
                self.data['hf_scf_energy'] = None
            else:
                self.data['hf_scf_energy'] = float(temp_dict['SCF'][0][0])
            if temp_dict.get('MP2') is None:
                self.data['mp2_energy'] = None
            else:
                self.data['mp2_energy'] = float(temp_dict['MP2'][0][0])
            if temp_dict.get('CCSD_correlation') is None:
                self.data['ccsd_correlation_energy'] = None
            else:
                self.data['ccsd_correlation_energy'] = float(temp_dict['CCSD_correlation'][0][0])
            if temp_dict.get('CCSD') is None:
                self.data['ccsd_total_energy'] = None
            else:
                self.data['ccsd_total_energy'] = float(temp_dict['CCSD'][0][0])
            if temp_dict.get('CCSD(T)_correlation') is None:
                self.data['ccsd(t)_correlation_energy'] = None
            else:
                self.data['ccsd(t)_correlation_energy'] = float(temp_dict['CCSD(T)_correlation'][0][0])
            if temp_dict.get('CCSD(T)') is None:
                self.data['ccsd(t)_total_energy'] = None
            else:
                self.data['ccsd(t)_total_energy'] = float(temp_dict['CCSD(T)'][0][0])
        self.data['optimization'] = read_pattern(self.text, {'key': '(?i)\\s*job(?:_)*type\\s*(?:=)*\\s*opt'}).get('key')
        if self.data.get('optimization', []):
            self.data['new_optimizer'] = read_pattern(self.text, {'key': '(?i)\\s*geom_opt2\\s*(?:=)*\\s*3'}).get('key')
            if self.data['version'] == '6':
                temp_driver = read_pattern(self.text, {'key': '(?i)\\s*geom_opt_driver\\s*(?:=)*\\s*optimize'}).get('key')
                if temp_driver is None:
                    self.data['new_optimizer'] = [[]]
            tmp_transition_state = read_pattern(self.text, {'key': 'TRANSITION STATE CONVERGED'}).get('key')
            if tmp_transition_state is not None:
                self.data['warnings']['unexpected_transition_state'] = True
            self._read_optimization_data()
        self.data['transition_state'] = read_pattern(self.text, {'key': '(?i)\\s*job(?:_)*type\\s*(?:=)*\\s*ts'}).get('key')
        if self.data.get('transition_state', []):
            self._read_optimization_data()
        self.data['opt_constraint'] = read_pattern(self.text, {'key': '\\$opt\\s+CONSTRAINT'}).get('key')
        if self.data.get('opt_constraint'):
            temp_constraint = read_pattern(self.text, {'key': 'Constraints and their Current Values\\s+Value\\s+Constraint\\s+(\\w+)\\:\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)'}).get('key')
            if temp_constraint is not None:
                self.data['opt_constraint'] = temp_constraint[0]
                if self.data.get('opt_constraint') is not None and float(self.data['opt_constraint'][5]) != float(self.data['opt_constraint'][6]):
                    if abs(float(self.data['opt_constraint'][5])) != abs(float(self.data['opt_constraint'][6])):
                        raise ValueError('ERROR: Opt section value and constraint should be the same!')
                    if abs(float(self.data['opt_constraint'][5])) not in [0.0, 180.0]:
                        raise ValueError('ERROR: Opt section value and constraint can only differ by a sign at 0.0 and 180.0!')
        self.data['frequency_job'] = read_pattern(self.text, {'key': '(?i)\\s*job(?:_)*type\\s*(?:=)*\\s*freq'}, terminate_on_match=True).get('key')
        if self.data.get('frequency_job', []):
            self._read_frequency_data()
        self.data['single_point_job'] = read_pattern(self.text, {'key': '(?i)\\s*job(?:_)*type\\s*(?:=)*\\s*sp'}, terminate_on_match=True).get('key')
        self.data['force_job'] = read_pattern(self.text, {'key': '(?i)\\s*job(?:_)*type\\s*(?:=)*\\s*force'}, terminate_on_match=True).get('key')
        if self.data.get('force_job', []):
            self._read_force_data()
        if self.data['scf_final_print'] >= 1:
            self._read_eigenvalues()
        if self.data['scf_final_print'] >= 3:
            self._read_fock_matrix()
            self._read_coefficient_matrix()
        self.data['scan_job'] = read_pattern(self.text, {'key': '(?i)\\s*job(?:_)*type\\s*(?:=)*\\s*pes_scan'}, terminate_on_match=True).get('key')
        if self.data.get('scan_job', []):
            self._read_scan_data()
        self.data['nbo_data'] = read_pattern(self.text, {'key': 'N A T U R A L   A T O M I C   O R B I T A L'}, terminate_on_match=True).get('key')
        if self.data.get('nbo_data', []):
            self._read_nbo_data()
        if not self.data.get('completion', []) and self.data.get('errors') == []:
            self._check_completion_errors()

    @staticmethod
    def multiple_outputs_from_file(filename, keep_sub_files=True):
        """
        Parses a QChem output file with multiple calculations
        # 1.) Separates the output into sub-files
            e.g. qcout -> qcout.0, qcout.1, qcout.2 ... qcout.N
            a.) Find delimiter for multiple calculations
            b.) Make separate output sub-files
        2.) Creates separate QCCalcs for each one from the sub-files.
        """
        to_return = []
        with zopen(filename, mode='rt') as file:
            text = re.split('\\s*(?:Running\\s+)*Job\\s+\\d+\\s+of\\s+\\d+\\s+', file.read())
        if text[0] == '':
            text = text[1:]
        for i, sub_text in enumerate(text):
            with open(f'{filename}.{i}', mode='w') as temp:
                temp.write(sub_text)
            tempOutput = QCOutput(f'{filename}.{i}')
            to_return.append(tempOutput)
            if not keep_sub_files:
                os.remove(f'{filename}.{i}')
        return to_return

    def _read_eigenvalues(self):
        """Parse the orbital energies from the output file. An array of the
        dimensions of the number of orbitals used in the calculation is stored.
        """
        header_pattern = 'Final Alpha MO Eigenvalues'
        elements_pattern = '\\-*\\d+\\.\\d+'
        if not self.data.get('unrestricted', []):
            spin_unrestricted = False
            footer_pattern = 'Final Alpha MO Coefficients+\\s*'
        else:
            spin_unrestricted = True
            footer_pattern = 'Final Beta MO Eigenvalues'
        alpha_eigenvalues = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
        if spin_unrestricted:
            header_pattern = 'Final Beta MO Eigenvalues'
            footer_pattern = 'Final Alpha MO Coefficients+\\s*'
            beta_eigenvalues = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
        self.data['alpha_eigenvalues'] = alpha_eigenvalues
        if spin_unrestricted:
            self.data['beta_eigenvalues'] = beta_eigenvalues

    def _read_fock_matrix(self):
        """Parses the Fock matrix. The matrix is read in whole
        from the output file and then transformed into the right dimensions.
        """
        header_pattern = 'Final Alpha Fock Matrix'
        elements_pattern = '\\-*\\d+\\.\\d+'
        if not self.data.get('unrestricted', []):
            spin_unrestricted = False
            footer_pattern = 'SCF time:'
        else:
            spin_unrestricted = True
            footer_pattern = 'Final Beta Fock Matrix'
        alpha_fock_matrix = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
        if spin_unrestricted:
            header_pattern = 'Final Beta Fock Matrix'
            footer_pattern = 'SCF time:'
            beta_fock_matrix = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
        alpha_fock_matrix = process_parsed_fock_matrix(alpha_fock_matrix)
        self.data['alpha_fock_matrix'] = alpha_fock_matrix
        if spin_unrestricted:
            beta_fock_matrix = process_parsed_fock_matrix(beta_fock_matrix)
            self.data['beta_fock_matrix'] = beta_fock_matrix

    def _read_coefficient_matrix(self):
        """Parses the coefficient matrix from the output file. Done is much
        the same was as the Fock matrix.
        """
        header_pattern = 'Final Alpha MO Coefficients'
        elements_pattern = '\\-*\\d+\\.\\d+'
        if not self.data.get('unrestricted', []):
            spin_unrestricted = False
            footer_pattern = 'Final Alpha Density Matrix'
        else:
            spin_unrestricted = True
            footer_pattern = 'Final Beta MO Coefficients'
        alpha_coeff_matrix = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
        if spin_unrestricted:
            header_pattern = 'Final Beta MO Coefficients'
            footer_pattern = 'Final Alpha Density Matrix'
            beta_coeff_matrix = read_matrix_pattern(header_pattern, footer_pattern, elements_pattern, self.text, postprocess=float)
        alpha_coeff_matrix = process_parsed_fock_matrix(alpha_coeff_matrix)
        self.data['alpha_coeff_matrix'] = alpha_coeff_matrix
        if spin_unrestricted:
            beta_coeff_matrix = process_parsed_fock_matrix(beta_coeff_matrix)
            self.data['beta_coeff_matrix'] = beta_coeff_matrix

    def _read_charge_and_multiplicity(self):
        """Parses charge and multiplicity."""
        temp_charge = read_pattern(self.text, {'key': '\\$molecule\\s+([\\-\\d]+)\\s+\\d'}, terminate_on_match=True).get('key')
        if temp_charge is not None:
            self.data['charge'] = int(temp_charge[0][0])
        else:
            temp_charge = read_pattern(self.text, {'key': 'Sum of atomic charges \\=\\s+([\\d\\-\\.\\+]+)'}, terminate_on_match=True).get('key')
            if temp_charge is None:
                self.data['charge'] = None
            else:
                self.data['charge'] = int(float(temp_charge[0][0]))
        temp_multiplicity = read_pattern(self.text, {'key': '\\$molecule\\s+[\\-\\d]+\\s+(\\d)'}, terminate_on_match=True).get('key')
        if temp_multiplicity is not None:
            self.data['multiplicity'] = int(temp_multiplicity[0][0])
        else:
            temp_multiplicity = read_pattern(self.text, {'key': 'Sum of spin\\s+charges \\=\\s+([\\d\\-\\.\\+]+)'}, terminate_on_match=True).get('key')
            if temp_multiplicity is None:
                self.data['multiplicity'] = 1
            else:
                self.data['multiplicity'] = int(float(temp_multiplicity[0][0])) + 1

    def _read_species_and_inital_geometry(self):
        """Parses species and initial geometry."""
        header_pattern = 'Standard Nuclear Orientation \\(Angstroms\\)\\s+I\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
        table_pattern = '\\s*\\d+\\s+([a-zA-Z]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*'
        footer_pattern = '\\s*-+'
        temp_geom = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
        if temp_geom is None or len(temp_geom) == 0:
            self.data['species'] = self.data['initial_geometry'] = self.data['initial_molecule'] = self.data['point_group'] = None
        else:
            temp_point_group = read_pattern(self.text, {'key': 'Molecular Point Group\\s+([A-Za-z\\d\\*]+)'}, terminate_on_match=True).get('key')
            if temp_point_group is not None:
                self.data['point_group'] = temp_point_group[0][0]
            else:
                self.data['point_group'] = None
            temp_geom = temp_geom[0]
            species = []
            geometry = np.zeros(shape=(len(temp_geom), 3), dtype=float)
            for ii, entry in enumerate(temp_geom):
                species += [entry[0]]
                for jj in range(3):
                    if '*' in entry[jj + 1]:
                        geometry[ii, jj] = 10000000000.0
                    else:
                        geometry[ii, jj] = float(entry[jj + 1])
            self.data['species'] = species
            self.data['initial_geometry'] = geometry
            if self.data['charge'] is not None and self.data['multiplicity'] is not None:
                self.data['initial_molecule'] = Molecule(species=species, coords=geometry, charge=self.data.get('charge'), spin_multiplicity=self.data.get('multiplicity'))
            else:
                self.data['initial_molecule'] = None

    def _read_SCF(self):
        """Parses both old and new SCFs."""
        if self.data.get('using_GEN_SCFMAN', []):
            footer_pattern = '(^\\s*\\-+\\n\\s+SCF time|^\\s*gen_scfman_exception: SCF failed to converge)'
            header_pattern = '^\\s*\\-+\\s+Cycle\\s+Energy\\s+(?:(?:DIIS)*\\s+[Ee]rror)*(?:RMS Gradient)*\\s+\\-+(?:\\s*\\-+\\s+OpenMP\\s+Integral\\s+computing\\s+Module\\s+(?:Release:\\s+version\\s+[\\d\\-\\.]+\\,\\s+\\w+\\s+[\\d\\-\\.]+\\, Q-Chem Inc\\. Pittsburgh\\s+)*\\-+)*\\n'
            table_pattern = '(?:\\s*Nonlocal correlation = [\\d\\-\\.]+e[\\d\\-]+)*(?:\\s*Inaccurate integrated density:\\n\\s+Number of electrons\\s+=\\s+[\\d\\-\\.]+\\n\\s+Numerical integral\\s+=\\s+[\\d\\-\\.]+\\n\\s+Relative error\\s+=\\s+[\\d\\-\\.]+\\s+\\%\\n)*\\s*\\d+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)e([\\d\\-\\.\\+]+)(?:\\s+Convergence criterion met)*(?:\\s+Preconditoned Steepest Descent)*(?:\\s+Roothaan Step)*(?:\\s+(?:Normal\\s+)*BFGS [Ss]tep)*(?:\\s+LineSearch Step)*(?:\\s+Line search: overstep)*(?:\\s+Dog-leg BFGS step)*(?:\\s+Line search: understep)*(?:\\s+Descent step)*(?:\\s+Done DIIS. Switching to GDM)*(?:\\s+Done GDM. Switching to DIIS)*(?:(?:\\s+Done GDM. Switching to GDM with quadratic line-search\\s)*\\s*GDM subspace size\\: \\d+)*(?:\\s*\\-+\\s+Cycle\\s+Energy\\s+(?:(?:DIIS)*\\s+[Ee]rror)*(?:RMS Gradient)*\\s+\\-+(?:\\s*\\-+\\s+OpenMP\\s+Integral\\s+computing\\s+Module\\s+(?:Release:\\s+version\\s+[\\d\\-\\.]+\\,\\s+\\w+\\s+[\\d\\-\\.]+\\, Q-Chem Inc\\. Pittsburgh\\s+)*\\-+)*\\n)*(?:\\s*Line search, dEdstep = [\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s*)*(?:\\s*[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+Optimal value differs by [\\d\\-\\.]+e[\\d\\-\\.\\+]+ from prediction)*(?:\\s*Resetting GDM\\.)*(?:\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+)*(?:\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+[\\d\\-\\.]+\\s+[\\d\\-\\.]+e[\\d\\-\\.\\+]+\\s+Optimal value differs by [\\d\\-\\.]+e[\\d\\-\\.\\+]+ from prediction)*(?:\\s*gdm_qls\\: Orbitals will not converge further\\.)*(?:(\\n\\s*[a-z\\dA-Z_\\s/]+\\.C|\\n\\s*GDM)::WARNING energy changes are now smaller than effective accuracy\\.\\s*(\\n\\s*[a-z\\dA-Z_\\s/]+\\.C|\\n\\s*GDM)::\\s+calculation will continue, but THRESH should be increased\\s*(\\n\\s*[a-z\\dA-Z_\\s/]+\\.C|\\n\\s*GDM)::\\s+or SCF_CONVERGENCE decreased\\.\\s*(\\n\\s*[a-z\\dA-Z_\\s/]+\\.C|\\n\\s*GDM)::\\s+effective_thresh = [\\d\\-\\.]+e[\\d\\-]+)*'
        else:
            if 'SCF_failed_to_converge' in self.data.get('errors'):
                footer_pattern = '^\\s*\\d+\\s*[\\d\\-\\.]+\\s+[\\d\\-\\.]+E[\\d\\-\\.]+\\s+Convergence\\s+failure\\n'
            else:
                footer_pattern = '^\\s*\\-+\\n'
            header_pattern = '^\\s*\\-+\\s+Cycle\\s+Energy\\s+DIIS Error\\s+\\-+\\n'
            table_pattern = '(?:\\s*Inaccurate integrated density:\\n\\s+Number of electrons\\s+=\\s+[\\d\\-\\.]+\\n\\s+Numerical integral\\s+=\\s+[\\d\\-\\.]+\\n\\s+Relative error\\s+=\\s+[\\d\\-\\.]+\\s+\\%\\n)*\\s*\\d+\\s*([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)E([\\d\\-\\.\\+]+)(?:\\s*\\n\\s*cpu\\s+[\\d\\-\\.]+\\swall\\s+[\\d\\-\\.]+)*(?:\\nin dftxc\\.C, eleTot sum is:[\\d\\-\\.]+, tauTot is\\:[\\d\\-\\.]+)*(?:\\s+Convergence criterion met)*(?:\\s+Done RCA\\. Switching to DIIS)*(?:\\n\\s*Warning: not using a symmetric Q)*(?:\\nRecomputing EXC\\s*[\\d\\-\\.]+\\s*[\\d\\-\\.]+\\s*[\\d\\-\\.]+(?:\\s*\\nRecomputing EXC\\s*[\\d\\-\\.]+\\s*[\\d\\-\\.]+\\s*[\\d\\-\\.]+)*)*'
        temp_scf = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
        real_scf = []
        for one_scf in temp_scf:
            temp = np.zeros(shape=(len(one_scf), 2))
            for ii, entry in enumerate(one_scf):
                temp[ii, 0] = float(entry[0])
                temp[ii, 1] = float(entry[1]) * 10 ** float(entry[2])
            real_scf += [temp]
        self.data['SCF'] = real_scf
        temp_thresh_warning = read_pattern(self.text, {'key': '\\n[a-zA-Z_\\s/]+\\.C::WARNING energy changes are now smaller than effective accuracy\\.\\n[a-zA-Z_\\s/]+\\.C::\\s+calculation will continue, but THRESH should be increased\\n[a-zA-Z_\\s/]+\\.C::\\s+or SCF_CONVERGENCE decreased\\. \\n[a-zA-Z_\\s/]+\\.C::\\s+effective_thresh = ([\\d\\-\\.]+e[\\d\\-]+)'}).get('key')
        if temp_thresh_warning is not None:
            if len(temp_thresh_warning) == 1:
                self.data['warnings']['thresh'] = float(temp_thresh_warning[0][0])
            else:
                thresh_warning = np.zeros(len(temp_thresh_warning))
                for ii, entry in enumerate(temp_thresh_warning):
                    thresh_warning[ii] = float(entry[0])
                self.data['warnings']['thresh'] = thresh_warning
        temp_SCF_energy = read_pattern(self.text, {'key': 'SCF   energy in the final basis set =\\s*([\\d\\-\\.]+)'}).get('key')
        if temp_SCF_energy is not None:
            if len(temp_SCF_energy) == 1:
                self.data['SCF_energy_in_the_final_basis_set'] = float(temp_SCF_energy[0][0])
            else:
                SCF_energy = np.zeros(len(temp_SCF_energy))
                for ii, val in enumerate(temp_SCF_energy):
                    SCF_energy[ii] = float(val[0])
                self.data['SCF_energy_in_the_final_basis_set'] = SCF_energy
        temp_Total_energy = read_pattern(self.text, {'key': 'Total energy in the final basis set =\\s*([\\d\\-\\.]+)'}).get('key')
        if temp_Total_energy is not None:
            if len(temp_Total_energy) == 1:
                self.data['Total_energy_in_the_final_basis_set'] = float(temp_Total_energy[0][0])
            else:
                Total_energy = np.zeros(len(temp_Total_energy))
                for ii, val in enumerate(temp_Total_energy):
                    Total_energy[ii] = float(val[0])
                self.data['Total_energy_in_the_final_basis_set'] = Total_energy

    def _read_charges_and_dipoles(self):
        """
        Parses Mulliken/ESP/RESP charges.
        Parses associated dipole/multipole moments.
        Also parses spins given an unrestricted SCF.
        """
        self.data['dipoles'] = {}
        temp_dipole_total = read_pattern(self.text, {'key': 'X\\s*[\\d\\-\\.]+\\s*Y\\s*[\\d\\-\\.]+\\s*Z\\s*[\\d\\-\\.]+\\s*Tot\\s*([\\d\\-\\.]+)'}).get('key')
        temp_dipole = read_pattern(self.text, {'key': 'X\\s*([\\d\\-\\.]+)\\s*Y\\s*([\\d\\-\\.]+)\\s*Z\\s*([\\d\\-\\.]+)\\s*Tot\\s*[\\d\\-\\.]+'}).get('key')
        if temp_dipole is not None:
            if len(temp_dipole_total) == 1:
                self.data['dipoles']['total'] = float(temp_dipole_total[0][0])
                dipole = np.zeros(3)
                for ii, val in enumerate(temp_dipole[0]):
                    dipole[ii] = float(val)
                self.data['dipoles']['dipole'] = dipole
            else:
                total = np.zeros(len(temp_dipole_total))
                for ii, val in enumerate(temp_dipole_total):
                    total[ii] = float(val[0])
                self.data['dipoles']['total'] = total
                dipole = np.zeros(shape=(len(temp_dipole_total), 3))
                for ii in range(len(temp_dipole)):
                    for jj, _val in enumerate(temp_dipole[ii]):
                        dipole[ii][jj] = temp_dipole[ii][jj]
                self.data['dipoles']['dipole'] = dipole
        self.data['multipoles'] = dict()
        quad_mom_pat = '\\s*Quadrupole Moments \\(Debye\\-Ang\\)\\s+XX\\s+([\\-\\.0-9]+)\\s+XY\\s+([\\-\\.0-9]+)\\s+YY\\s+([\\-\\.0-9]+)\\s+XZ\\s+([\\-\\.0-9]+)\\s+YZ\\s+([\\-\\.0-9]+)\\s+ZZ\\s+([\\-\\.0-9]+)'
        temp_quadrupole_moment = read_pattern(self.text, {'key': quad_mom_pat}).get('key')
        if temp_quadrupole_moment is not None:
            keys = ('XX', 'XY', 'YY', 'XZ', 'YZ', 'ZZ')
            if len(temp_quadrupole_moment) == 1:
                self.data['multipoles']['quadrupole'] = {key: float(temp_quadrupole_moment[0][idx]) for idx, key in enumerate(keys)}
            else:
                self.data['multipoles']['quadrupole'] = list()
                for qpole in temp_quadrupole_moment:
                    self.data['multipoles']['quadrupole'].append({key: float(qpole[idx]) for idx, key in enumerate(keys)})
        octo_mom_pat = '\\s*Octopole Moments \\(Debye\\-Ang\\^2\\)\\s+XXX\\s+([\\-\\.0-9]+)\\s+XXY\\s+([\\-\\.0-9]+)\\s+XYY\\s+([\\-\\.0-9]+)\\s+YYY\\s+([\\-\\.0-9]+)\\s+XXZ\\s+([\\-\\.0-9]+)\\s+XYZ\\s+([\\-\\.0-9]+)\\s+YYZ\\s+([\\-\\.0-9]+)\\s+XZZ\\s+([\\-\\.0-9]+)\\s+YZZ\\s+([\\-\\.0-9]+)\\s+ZZZ\\s+([\\-\\.0-9]+)'
        temp_octopole_moment = read_pattern(self.text, {'key': octo_mom_pat}).get('key')
        if temp_octopole_moment is not None:
            keys = ('XXX', 'XXY', 'XYY', 'YYY', 'XXZ', 'XYZ', 'YYZ', 'XZZ', 'YZZ', 'ZZZ')
            if len(temp_octopole_moment) == 1:
                self.data['multipoles']['octopole'] = {key: float(temp_octopole_moment[0][idx]) for idx, key in enumerate(keys)}
            else:
                self.data['multipoles']['octopole'] = list()
                for opole in temp_octopole_moment:
                    self.data['multipoles']['octopole'].append({key: float(opole[idx]) for idx, key in enumerate(keys)})
        hexadeca_mom_pat = '\\s*Hexadecapole Moments \\(Debye\\-Ang\\^3\\)\\s+XXXX\\s+([\\-\\.0-9]+)\\s+XXXY\\s+([\\-\\.0-9]+)\\s+XXYY\\s+([\\-\\.0-9]+)\\s+XYYY\\s+([\\-\\.0-9]+)\\s+YYYY\\s+([\\-\\.0-9]+)\\s+XXXZ\\s+([\\-\\.0-9]+)\\s+XXYZ\\s+([\\-\\.0-9]+)\\s+XYYZ\\s+([\\-\\.0-9]+)\\s+YYYZ\\s+([\\-\\.0-9]+)\\s+XXZZ\\s+([\\-\\.0-9]+)\\s+XYZZ\\s+([\\-\\.0-9]+)\\s+YYZZ\\s+([\\-\\.0-9]+)\\s+XZZZ\\s+([\\-\\.0-9]+)\\s+YZZZ\\s+([\\-\\.0-9]+)\\s+ZZZZ\\s+([\\-\\.0-9]+)'
        temp_hexadecapole_moment = read_pattern(self.text, {'key': hexadeca_mom_pat}).get('key')
        if temp_hexadecapole_moment is not None:
            keys = ('XXXX', 'XXXY', 'XXYY', 'XYYY', 'YYYY', 'XXXZ', 'XXYZ', 'XYYZ', 'YYYZ', 'XXZZ', 'XYZZ', 'YYZZ', 'XZZZ', 'YZZZ', 'ZZZZ')
            if len(temp_hexadecapole_moment) == 1:
                self.data['multipoles']['hexadecapole'] = {key: float(temp_hexadecapole_moment[0][idx]) for idx, key in enumerate(keys)}
            else:
                self.data['multipoles']['hexadecapole'] = list()
                for hpole in temp_hexadecapole_moment:
                    self.data['multipoles']['hexadecapole'].append({key: float(hpole[idx]) for idx, key in enumerate(keys)})
        if self.data.get('unrestricted', []):
            header_pattern = '\\-+\\s+Ground-State Mulliken Net Atomic Charges\\s+Atom\\s+Charge \\(a\\.u\\.\\)\\s+Spin\\s\\(a\\.u\\.\\)\\s+\\-+'
            table_pattern = '\\s+\\d+\\s\\w+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)'
            footer_pattern = '\\s\\s\\-+\\s+Sum of atomic charges'
        else:
            header_pattern = '\\-+\\s+Ground-State Mulliken Net Atomic Charges\\s+Atom\\s+Charge \\(a\\.u\\.\\)\\s+\\-+'
            table_pattern = '\\s+\\d+\\s\\w+\\s+([\\d\\-\\.]+)'
            footer_pattern = '\\s\\s\\-+\\s+Sum of atomic charges'
        temp_mulliken = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
        real_mulliken = []
        for one_mulliken in temp_mulliken:
            if self.data.get('unrestricted', []):
                temp = np.zeros(shape=(len(one_mulliken), 2))
                for ii, entry in enumerate(one_mulliken):
                    temp[ii, 0] = float(entry[0])
                    temp[ii, 1] = float(entry[1])
            else:
                temp = np.zeros(len(one_mulliken))
                for ii, entry in enumerate(one_mulliken):
                    temp[ii] = float(entry[0])
            real_mulliken += [temp]
        self.data['Mulliken'] = real_mulliken
        esp_or_resp = read_pattern(self.text, {'key': 'Merz-Kollman (R?ESP) Net Atomic Charges'}).get('key')
        if esp_or_resp is not None:
            header_pattern = 'Merz-Kollman (R?ESP) Net Atomic Charges\\s+Atom\\s+Charge \\(a\\.u\\.\\)\\s+\\-+'
            table_pattern = '\\s+\\d+\\s\\w+\\s+([\\d\\-\\.]+)'
            footer_pattern = '\\s\\s\\-+\\s+Sum of atomic charges'
            temp_esp_or_resp = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
            real_esp_or_resp = []
            for one_entry in temp_esp_or_resp:
                temp = np.zeros(len(one_entry))
                for ii, entry in enumerate(one_entry):
                    temp[ii] = float(entry[0])
                real_esp_or_resp += [temp]
            self.data[esp_or_resp[0][0]] = real_esp_or_resp
            temp_RESP_dipole_total = read_pattern(self.text, {'key': 'Related Dipole Moment =\\s*([\\d\\-\\.]+)\\s*\\(X\\s*[\\d\\-\\.]+\\s*Y\\s*[\\d\\-\\.]+\\s*Z\\s*[\\d\\-\\.]+\\)'}).get('key')
            temp_RESP_dipole = read_pattern(self.text, {'key': 'Related Dipole Moment =\\s*[\\d\\-\\.]+\\s*\\(X\\s*([\\d\\-\\.]+)\\s*Y\\s*([\\d\\-\\.]+)\\s*Z\\s*([\\d\\-\\.]+)\\)'}).get('key')
            if temp_RESP_dipole is not None:
                if len(temp_RESP_dipole_total) == 1:
                    self.data['dipoles']['RESP_total'] = float(temp_RESP_dipole_total[0][0])
                    RESP_dipole = np.zeros(3)
                    for ii, val in enumerate(temp_RESP_dipole[0]):
                        RESP_dipole[ii] = float(val)
                    self.data['dipoles']['RESP_dipole'] = RESP_dipole
                else:
                    RESP_total = np.zeros(len(temp_RESP_dipole_total))
                    for ii, val in enumerate(temp_RESP_dipole_total):
                        RESP_total[ii] = float(val[0])
                    self.data['dipoles']['RESP_total'] = RESP_total
                    RESP_dipole = np.zeros(shape=(len(temp_RESP_dipole_total), 3))
                    for ii in range(len(temp_RESP_dipole)):
                        for jj, _val in enumerate(temp_RESP_dipole[ii]):
                            RESP_dipole[ii][jj] = temp_RESP_dipole[ii][jj]
                    self.data['dipoles']['RESP_dipole'] = RESP_dipole

    def _detect_general_warnings(self):
        temp_inac_integ = read_pattern(self.text, {'key': 'Inaccurate integrated density:\\n\\s+Number of electrons\\s+=\\s+([\\d\\-\\.]+)\\n\\s+Numerical integral\\s+=\\s+([\\d\\-\\.]+)\\n\\s+Relative error\\s+=\\s+([\\d\\-\\.]+)\\s+\\%\\n'}).get('key')
        if temp_inac_integ is not None:
            inaccurate_integrated_density = np.zeros(shape=(len(temp_inac_integ), 3))
            for ii, entry in enumerate(temp_inac_integ):
                for jj, val in enumerate(entry):
                    inaccurate_integrated_density[ii][jj] = float(val)
            self.data['warnings']['inaccurate_integrated_density'] = inaccurate_integrated_density
        if read_pattern(self.text, {'key': 'Intel MKL ERROR'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['mkl'] = True
        if read_pattern(self.text, {'key': 'Starting finite difference calculation for IDERIV'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['missing_analytical_derivates'] = True
        if read_pattern(self.text, {'key': 'Inconsistent size for SCF MO coefficient file'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['inconsistent_size'] = True
        if read_pattern(self.text, {'key': 'Linear dependence detected in AO basis'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['linear_dependence'] = True
        if read_pattern(self.text, {'key': '\\*\\*WARNING\\*\\* Hessian does not have the Desired Local Structure'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['hessian_local_structure'] = True
        if read_pattern(self.text, {'key': '\\*\\*\\*ERROR\\*\\*\\* Exceeded allowed number of iterative cycles in GetCART'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['GetCART_cycles'] = True
        if read_pattern(self.text, {'key': '\\*\\*WARNING\\*\\* Problems with Internal Coordinates'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['internal_coordinates'] = True
        if read_pattern(self.text, {'key': 'UNABLE TO DETERMINE Lambda IN RFO  \\*\\*\\s+\\*\\* Taking simple Newton-Raphson step'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['bad_lambda_take_NR_step'] = True
        if read_pattern(self.text, {'key': 'SWITCHING TO CARTESIAN OPTIMIZATION'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['switch_to_cartesian'] = True
        if read_pattern(self.text, {'key': '\\*\\*WARNING\\*\\* Magnitude of eigenvalue'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['eigenvalue_magnitude'] = True
        if read_pattern(self.text, {'key': '\\*\\*WARNING\\*\\* Hereditary positive definiteness endangered'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['positive_definiteness_endangered'] = True
        if read_pattern(self.text, {'key': '\\*\\*\\*ERROR\\*\\*\\* Angle[\\s\\d]+is near\\-linear\\s+But No atom available to define colinear bend'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['colinear_bend'] = True
        if read_pattern(self.text, {'key': '\\*\\*\\*ERROR\\*\\*\\* Unable to Diagonalize B\\*B\\(t\\) in <MakeNIC>'}, terminate_on_match=True).get('key') == [[]]:
            self.data['warnings']['diagonalizing_BBt'] = True

    def _read_geometries(self):
        """Parses all geometries from an optimization trajectory."""
        geoms = []
        if self.data.get('new_optimizer') is None:
            header_pattern = '\\s+Optimization\\sCycle:\\s+\\d+\\s+Coordinates \\(Angstroms\\)\\s+ATOM\\s+X\\s+Y\\s+Z'
            table_pattern = '\\s+\\d+\\s+\\w+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)'
            footer_pattern = '\\s+Point Group\\:\\s+[\\d\\w\\*]+\\s+Number of degrees of freedom\\:\\s+\\d+'
        elif read_pattern(self.text, {'key': 'Geometry Optimization Coordinates :\\s+Cartesian'}, terminate_on_match=True).get('key') == [[]]:
            header_pattern = 'RMS\\s+of Stepsize\\s+[\\d\\-\\.]+\\s+-+\\s+Standard Nuclear Orientation \\(Angstroms\\)\\s+I\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
            table_pattern = '\\s*\\d+\\s+[a-zA-Z]+\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*'
            footer_pattern = '\\s*-+'
        else:
            header_pattern = 'Finished Iterative Coordinate Back-Transformation\\s+-+\\s+Standard Nuclear Orientation \\(Angstroms\\)\\s+I\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
            table_pattern = '\\s*\\d+\\s+[a-zA-Z]+\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*'
            footer_pattern = '\\s*-+'
        parsed_geometries = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
        for parsed_geometry in parsed_geometries:
            if not parsed_geometry:
                geoms.append(None)
            else:
                geoms.append(process_parsed_coords(parsed_geometry))
        if len(geoms) >= 1:
            self.data['geometries'] = geoms
            self.data['last_geometry'] = geoms[-1]
            if self.data.get('charge') is not None:
                self.data['molecule_from_last_geometry'] = Molecule(species=self.data.get('species'), coords=self.data.get('last_geometry'), charge=self.data.get('charge'), spin_multiplicity=self.data.get('multiplicity'))
            if self.data.get('new_optimizer') is None:
                header_pattern = '\\*+\\s+(OPTIMIZATION|TRANSITION STATE)\\s+CONVERGED\\s+\\*+\\s+\\*+\\s+Coordinates \\(Angstroms\\)\\s+ATOM\\s+X\\s+Y\\s+Z'
                table_pattern = '\\s+\\d+\\s+\\w+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)'
                footer_pattern = '\\s+Z-matrix Print:'
            else:
                header_pattern = '(OPTIMIZATION|TRANSITION STATE)\\sCONVERGED\\s+\\*+\\s+\\*+\\s+-+\\s+Standard Nuclear Orientation \\(Angstroms\\)\\s+I\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
                table_pattern = '\\s*\\d+\\s+[a-zA-Z]+\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*'
                footer_pattern = '\\s*-+'
            parsed_optimized_geometries = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
            if not parsed_optimized_geometries:
                self.data['optimized_geometry'] = None
                header_pattern = '^\\s+\\*+\\s+(OPTIMIZATION|TRANSITION STATE) CONVERGED\\s+\\*+\\s+\\*+\\s+Z-matrix\\s+Print:\\s+\\$molecule\\s+[\\d\\-]+\\s+[\\d\\-]+\\n'
                table_pattern = '\\s*(\\w+)(?:\\s+(\\d+)\\s+([\\d\\-\\.]+)(?:\\s+(\\d+)\\s+([\\d\\-\\.]+)(?:\\s+(\\d+)\\s+([\\d\\-\\.]+))*)*)*(?:\\s+0)*'
                footer_pattern = '^\\$end\\n'
                self.data['optimized_zmat'] = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
            else:
                self.data['optimized_geometry'] = process_parsed_coords(parsed_optimized_geometries[0])
                self.data['optimized_geometries'] = [process_parsed_coords(i) for i in parsed_optimized_geometries]
                if self.data.get('charge') is not None:
                    self.data['molecule_from_optimized_geometry'] = Molecule(species=self.data.get('species'), coords=self.data.get('optimized_geometry'), charge=self.data.get('charge'), spin_multiplicity=self.data.get('multiplicity'))
                    self.data['molecules_from_optimized_geometries'] = []
                    for geom in self.data['optimized_geometries']:
                        mol = Molecule(species=self.data.get('species'), coords=geom, charge=self.data.get('charge'), spin_multiplicity=self.data.get('multiplicity'))
                        self.data['molecules_from_optimized_geometries'].append(mol)

    def _get_grad_format_length(self, header):
        """
        Determines the maximum number of gradient entries printed on a line,
        which changes for different versions of Q-Chem.
        """
        found_end = False
        index = 1
        pattern = header
        while not found_end:
            if read_pattern(self.text, {'key': pattern}, terminate_on_match=True).get('key') != [[]]:
                found_end = True
            else:
                pattern = f'{pattern}\\s+{index}'
                index += 1
        return index - 2

    def _read_gradients(self):
        """Parses all gradients obtained during an optimization trajectory."""
        grad_header_pattern = 'Gradient of (?:SCF)?(?:MP2)? Energy(?: \\(in au\\.\\))?'
        footer_pattern = '(?:Max gradient component|Gradient time)'
        grad_format_length = self._get_grad_format_length(grad_header_pattern)
        grad_table_pattern = '(?:\\s+\\d+(?:\\s+\\d+)?(?:\\s+\\d+)?(?:\\s+\\d+)?(?:\\s+\\d+)?(?:\\s+\\d+)?)?\\n\\s\\s\\s\\s[1-3]\\s*(\\-?[\\d\\.]{9,12})'
        if grad_format_length > 1:
            for _ in range(1, grad_format_length):
                grad_table_pattern = grad_table_pattern + '(?:\\s*(\\-?[\\d\\.]{9,12}))?'
        parsed_gradients = read_table_pattern(self.text, grad_header_pattern, grad_table_pattern, footer_pattern)
        if len(parsed_gradients) >= 1:
            sorted_gradients = np.zeros(shape=(len(parsed_gradients), len(self.data['initial_molecule']), 3))
            for ii, grad in enumerate(parsed_gradients):
                for jj in range(int(len(grad) / 3)):
                    for kk in range(grad_format_length):
                        if grad[jj * 3][kk] != 'None':
                            sorted_gradients[ii][jj * grad_format_length + kk][0] = grad[jj * 3][kk]
                            sorted_gradients[ii][jj * grad_format_length + kk][1] = grad[jj * 3 + 1][kk]
                            sorted_gradients[ii][jj * grad_format_length + kk][2] = grad[jj * 3 + 2][kk]
            self.data['gradients'] = sorted_gradients
            if self.data['solvent_method'] is not None:
                header_pattern = 'total gradient after adding PCM contribution --\\s+-+\\s+Atom\\s+X\\s+Y\\s+Z\\s+-+'
                table_pattern = '\\s+\\d+\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s+([\\d\\-\\.]+)\\s'
                footer_pattern = '-+'
                parsed_gradients = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
                pcm_gradients = np.zeros(shape=(len(parsed_gradients), len(self.data['initial_molecule']), 3))
                for ii, grad in enumerate(parsed_gradients):
                    for jj, entry in enumerate(grad):
                        for kk, val in enumerate(entry):
                            pcm_gradients[ii][jj][kk] = float(val)
                self.data['pcm_gradients'] = pcm_gradients
            else:
                self.data['pcm_gradients'] = None
            if read_pattern(self.text, {'key': 'Gradient of CDS energy'}, terminate_on_match=True).get('key') == [[]]:
                header_pattern = 'Gradient of CDS energy'
                parsed_gradients = read_table_pattern(self.text, header_pattern, grad_table_pattern, grad_header_pattern)
                sorted_gradients = np.zeros(shape=(len(parsed_gradients), len(self.data['initial_molecule']), 3))
                for ii, grad in enumerate(parsed_gradients):
                    for jj in range(int(len(grad) / 3)):
                        for kk in range(grad_format_length):
                            if grad[jj * 3][kk] != 'None':
                                sorted_gradients[ii][jj * grad_format_length + kk][0] = grad[jj * 3][kk]
                                sorted_gradients[ii][jj * grad_format_length + kk][1] = grad[jj * 3 + 1][kk]
                                sorted_gradients[ii][jj * grad_format_length + kk][2] = grad[jj * 3 + 2][kk]
                self.data['CDS_gradients'] = sorted_gradients
            else:
                self.data['CDS_gradients'] = None

    def _read_optimization_data(self):
        if self.data.get('new_optimizer') is None or self.data['version'] == '6':
            temp_energy_trajectory = read_pattern(self.text, {'key': '\\sEnergy\\sis\\s+([\\d\\-\\.]+)'}).get('key')
        else:
            temp_energy_trajectory = read_pattern(self.text, {'key': '\\sStep\\s*\\d+\\s*:\\s*Energy\\s*([\\d\\-\\.]+)'}).get('key')
        if self.data.get('new_optimizer') == [[]] and temp_energy_trajectory is not None:
            temp_energy_trajectory.insert(0, [str(self.data['Total_energy_in_the_final_basis_set'][0])])
        self._read_geometries()
        self._read_gradients()
        if temp_energy_trajectory is None:
            self.data['energy_trajectory'] = []
            if read_pattern(self.text, {'key': 'Error in back_transform'}, terminate_on_match=True).get('key') == [[]]:
                self.data['errors'] += ['back_transform_error']
            elif read_pattern(self.text, {'key': 'pinv\\(\\)\\: svd failed'}, terminate_on_match=True).get('key') == [[]]:
                self.data['errors'] += ['svd_failed']
        else:
            real_energy_trajectory = np.zeros(len(temp_energy_trajectory))
            for ii, entry in enumerate(temp_energy_trajectory):
                real_energy_trajectory[ii] = float(entry[0])
            self.data['energy_trajectory'] = real_energy_trajectory
            if self.data.get('new_optimizer') == [[]]:
                temp_norms = read_pattern(self.text, {'key': 'Norm of Stepsize\\s*([\\d\\-\\.]+)'}).get('key')
                if temp_norms is not None:
                    norms = np.zeros(len(temp_norms))
                    for ii, val in enumerate(temp_norms):
                        norms[ii] = float(val[0])
                    self.data['norm_of_stepsize'] = norms
            if openbabel is not None:
                self.data['structure_change'] = check_for_structure_changes(self.data['initial_molecule'], self.data['molecule_from_last_geometry'])
            if len(self.data.get('errors')) == 0 and self.data.get('optimized_geometry') is None and (len(self.data.get('optimized_zmat')) == 0):
                if read_pattern(self.text, {'key': 'MAXIMUM OPTIMIZATION CYCLES REACHED'}, terminate_on_match=True).get('key') == [[]] or read_pattern(self.text, {'key': 'Maximum number of iterations reached during minimization algorithm'}, terminate_on_match=True).get('key') == [[]]:
                    self.data['errors'] += ['out_of_opt_cycles']
                elif read_pattern(self.text, {'key': 'UNABLE TO DETERMINE Lamda IN FormD'}, terminate_on_match=True).get('key') == [[]]:
                    self.data['errors'] += ['unable_to_determine_lamda']
                elif read_pattern(self.text, {'key': 'Error in back_transform'}, terminate_on_match=True).get('key') == [[]]:
                    self.data['errors'] += ['back_transform_error']
                elif read_pattern(self.text, {'key': 'pinv\\(\\)\\: svd failed'}, terminate_on_match=True).get('key') == [[]]:
                    self.data['errors'] += ['svd_failed']

    def _read_frequency_data(self):
        """Parses cpscf_nseg, frequencies, enthalpy, entropy, and mode vectors."""
        if read_pattern(self.text, {'key': 'Calculating MO derivatives via CPSCF'}, terminate_on_match=True).get('key') == [[]]:
            temp_cpscf_nseg = read_pattern(self.text, {'key': 'CPSCF will be done in([\\d\\s]+)segments to save memory'}, terminate_on_match=True).get('key')
            if temp_cpscf_nseg is None:
                self.data['cpscf_nseg'] = 1
            else:
                self.data['cpscf_nseg'] = int(temp_cpscf_nseg[0][0])
        else:
            self.data['cpscf_nseg'] = 0
        raman = False
        if read_pattern(self.text, {'key': 'doraman\\s*(?:=)*\\s*true'}, terminate_on_match=True).get('key') == [[]]:
            raman = True
        temp_dict = read_pattern(self.text, {'frequencies': '\\s*Frequency:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+))*)*', 'trans_dip': 'TransDip\\s+(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(?:(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(?:(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7})\\s*(\\-?[\\d\\.]{5,7}|\\*{5,7}))*)*', 'IR_intens': '\\s*IR Intens:\\s*(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+))*)*', 'IR_active': '\\s*IR Active:\\s+([YESNO]+)(?:\\s+([YESNO]+)(?:\\s+([YESNO]+))*)*', 'raman_intens': '\\s*Raman Intens:\\s*(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+))*)*', 'depolar': '\\s*Depolar:\\s*(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+)(?:\\s+(\\-?[\\d\\.\\*]+))*)*', 'raman_active': '\\s*Raman Active:\\s+([YESNO]+)(?:\\s+([YESNO]+)(?:\\s+([YESNO]+))*)*', 'ZPE': '\\s*Zero point vibrational energy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'trans_enthalpy': '\\s*Translational Enthalpy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'rot_enthalpy': '\\s*Rotational Enthalpy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'vib_enthalpy': '\\s*Vibrational Enthalpy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'gas_constant': '\\s*gas constant \\(RT\\):\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'trans_entropy': '\\s*Translational Entropy:\\s+([\\d\\-\\.]+)\\s+cal/mol\\.K', 'rot_entropy': '\\s*Rotational Entropy:\\s+([\\d\\-\\.]+)\\s+cal/mol\\.K', 'vib_entropy': '\\s*Vibrational Entropy:\\s+([\\d\\-\\.]+)\\s+cal/mol\\.K', 'total_enthalpy': '\\s*Total Enthalpy:\\s+([\\d\\-\\.]+)\\s+kcal/mol', 'total_entropy': '\\s*Total Entropy:\\s+([\\d\\-\\.]+)\\s+cal/mol\\.K'})
        keys = ['ZPE', 'trans_enthalpy', 'rot_enthalpy', 'vib_enthalpy', 'gas_constant', 'trans_entropy', 'rot_entropy', 'vib_entropy', 'total_enthalpy', 'total_entropy']
        for key in keys:
            if temp_dict.get(key) is None:
                self.data[key] = None
            else:
                self.data[key] = float(temp_dict.get(key)[0][0])
        if temp_dict.get('frequencies') is None:
            self.data['frequencies'] = self.data['IR_intens'] = self.data['IR_active'] = None
            self.data['raman_active'] = self.data['raman_intens'] = None
            self.data['depolar'] = self.data['trans_dip'] = None
        else:
            temp_freqs = [value for entry in temp_dict.get('frequencies') for value in entry]
            temp_IR_intens = [value for entry in temp_dict.get('IR_intens') for value in entry]
            IR_active = [value for entry in temp_dict.get('IR_active') for value in entry]
            temp_trans_dip = [value for entry in temp_dict.get('trans_dip') for value in entry]
            self.data['IR_active'] = IR_active
            if raman:
                raman_active = [value for entry in temp_dict.get('raman_active') for value in entry]
                temp_raman_intens = [value for entry in temp_dict.get('raman_intens') for value in entry]
                temp_depolar = [value for entry in temp_dict.get('depolar') for value in entry]
                self.data['raman_active'] = raman_active
                raman_intens = np.zeros(len(temp_raman_intens) - temp_raman_intens.count('None'))
                for ii, entry in enumerate(temp_raman_intens):
                    if entry != 'None':
                        if '*' in entry:
                            raman_intens[ii] = float('inf')
                        else:
                            raman_intens[ii] = float(entry)
                self.data['raman_intens'] = raman_intens
                depolar = np.zeros(len(temp_depolar) - temp_depolar.count('None'))
                for ii, entry in enumerate(temp_depolar):
                    if entry != 'None':
                        if '*' in entry:
                            depolar[ii] = float('inf')
                        else:
                            depolar[ii] = float(entry)
                self.data['depolar'] = depolar
            else:
                self.data['raman_intens'] = self.data['raman_active'] = self.data['depolar'] = None
            trans_dip = np.zeros(shape=(int((len(temp_trans_dip) - temp_trans_dip.count('None')) / 3), 3))
            for ii, entry in enumerate(temp_trans_dip):
                if entry != 'None':
                    if '*' in entry:
                        trans_dip[int(ii / 3)][ii % 3] = float('inf')
                    else:
                        trans_dip[int(ii / 3)][ii % 3] = float(entry)
            self.data['trans_dip'] = trans_dip
            freqs = np.zeros(len(temp_freqs) - temp_freqs.count('None'))
            for ii, entry in enumerate(temp_freqs):
                if entry != 'None':
                    if '*' in entry:
                        if ii == 0:
                            freqs[ii] = -float('inf')
                        elif ii == len(freqs) - 1:
                            freqs[ii] = float('inf')
                        elif freqs[ii - 1] == -float('inf'):
                            freqs[ii] = -float('inf')
                        elif '*' in temp_freqs[ii + 1]:
                            freqs[ii] = float('inf')
                        else:
                            raise RuntimeError('ERROR: Encountered an undefined frequency not at the beginning or end of the frequency list, which makes no sense! Exiting...')
                        if not self.data.get('completion', []):
                            if 'undefined_frequency' not in self.data['errors']:
                                self.data['errors'] += ['undefined_frequency']
                        elif 'undefined_frequency' not in self.data['warnings']:
                            self.data['warnings']['undefined_frequency'] = True
                    else:
                        freqs[ii] = float(entry)
            self.data['frequencies'] = freqs
            IR_intens = np.zeros(len(temp_IR_intens) - temp_IR_intens.count('None'))
            for ii, entry in enumerate(temp_IR_intens):
                if entry != 'None':
                    if '*' in entry:
                        IR_intens[ii] = float('inf')
                    else:
                        IR_intens[ii] = float(entry)
            self.data['IR_intens'] = IR_intens
            if not raman:
                header_pattern = '\\s*Raman Active:\\s+[YESNO]+\\s+(?:[YESNO]+\\s+)*X\\s+Y\\s+Z\\s+(?:X\\s+Y\\s+Z\\s+)*'
            else:
                header_pattern = '\\s*Depolar:\\s*\\-?[\\d\\.\\*]+\\s+(?:\\-?[\\d\\.\\*]+\\s+)*X\\s+Y\\s+Z\\s+(?:X\\s+Y\\s+Z\\s+)*'
            table_pattern = '\\s*[a-zA-Z][a-zA-Z\\s]\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*(?:([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*(?:([\\d\\-\\.]+)\\s*([\\d\\-\\.]+)\\s*([\\d\\-\\.]+))*)*'
            footer_pattern = 'TransDip\\s+\\-?[\\d\\.\\*]+\\s*\\-?[\\d\\.\\*]+\\s*\\-?[\\d\\.\\*]+\\s*(?:\\-?[\\d\\.\\*]+\\s*\\-?[\\d\\.\\*]+\\s*\\-?[\\d\\.\\*]+\\s*)*'
            temp_freq_mode_vecs = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
            freq_mode_vecs = np.zeros(shape=(len(freqs), len(temp_freq_mode_vecs[0]), 3))
            for ii, triple_FMV in enumerate(temp_freq_mode_vecs):
                for jj, line in enumerate(triple_FMV):
                    for kk, entry in enumerate(line):
                        if entry != 'None':
                            freq_mode_vecs[int(ii * 3 + math.floor(kk / 3)), jj, kk % 3] = float(entry)
            self.data['frequency_mode_vectors'] = freq_mode_vecs
            freq_length = len(self.data['frequencies'])
            if len(self.data['frequency_mode_vectors']) != freq_length or len(self.data['IR_intens']) != freq_length or len(self.data['IR_active']) != freq_length:
                self.data['warnings']['frequency_length_inconsistency'] = True

    def _read_force_data(self):
        self._read_gradients()

    def _read_scan_data(self):
        temp_energy_trajectory = read_pattern(self.text, {'key': '\\sEnergy\\sis\\s+([\\d\\-\\.]+)'}).get('key')
        if temp_energy_trajectory is None:
            self.data['energy_trajectory'] = []
        else:
            real_energy_trajectory = np.zeros(len(temp_energy_trajectory))
            for ii, entry in enumerate(temp_energy_trajectory):
                real_energy_trajectory[ii] = float(entry[0])
            self.data['energy_trajectory'] = real_energy_trajectory
        self._read_geometries()
        if openbabel is not None:
            self.data['structure_change'] = check_for_structure_changes(self.data['initial_molecule'], self.data['molecule_from_last_geometry'])
        self._read_gradients()
        if len(self.data.get('errors')) == 0:
            if read_pattern(self.text, {'key': 'MAXIMUM OPTIMIZATION CYCLES REACHED'}, terminate_on_match=True).get('key') == [[]]:
                self.data['errors'] += ['out_of_opt_cycles']
            elif read_pattern(self.text, {'key': 'UNABLE TO DETERMINE Lamda IN FormD'}, terminate_on_match=True).get('key') == [[]]:
                self.data['errors'] += ['unable_to_determine_lamda']
        header_pattern = '\\s*\\-+ Summary of potential scan\\: \\-+\\s*'
        row_pattern_single = '\\s*([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n'
        row_pattern_double = '\\s*([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n'
        footer_pattern = '\\s*\\-+'
        single_data = read_table_pattern(self.text, header_pattern=header_pattern, row_pattern=row_pattern_single, footer_pattern=footer_pattern)
        self.data['scan_energies'] = []
        if len(single_data) == 0:
            double_data = read_table_pattern(self.text, header_pattern=header_pattern, row_pattern=row_pattern_double, footer_pattern=footer_pattern)
            if len(double_data) == 0:
                self.data['scan_energies'] = None
            else:
                for line in double_data[0]:
                    params = [float(line[0]), float(line[1])]
                    energy = float(line[2])
                    self.data['scan_energies'].append({'params': params, 'energy': energy})
        else:
            for line in single_data[0]:
                param = float(line[0])
                energy = float(line[1])
                self.data['scan_energies'].append({'params': param, 'energy': energy})
        scan_inputs_head = '\\s*\\$[Ss][Cc][Aa][Nn]'
        scan_inputs_row = '\\s*([Ss][Tt][Rr][Ee]|[Tt][Oo][Rr][Ss]|[Bb][Ee][Nn][Dd]) '
        scan_inputs_row += '((?:[0-9]+\\s+)+)([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*'
        scan_inputs_foot = '\\s*\\$[Ee][Nn][Dd]'
        constraints_meta = read_table_pattern(self.text, header_pattern=scan_inputs_head, row_pattern=scan_inputs_row, footer_pattern=scan_inputs_foot)
        self.data['scan_variables'] = {'stre': [], 'bend': [], 'tors': []}
        for row in constraints_meta[0]:
            var_type = row[0].lower()
            self.data['scan_variables'][var_type].append({'atoms': [int(i) for i in row[1].split()], 'start': float(row[2]), 'end': float(row[3]), 'increment': float(row[4])})
        temp_constraint = read_pattern(self.text, {'key': '\\s*(Distance\\(Angs\\)|Angle|Dihedral)\\:\\s*((?:[0-9]+\\s+)+)+([\\.0-9]+)\\s+([\\.0-9]+)'}).get('key')
        self.data['scan_constraint_sets'] = {'stre': [], 'bend': [], 'tors': []}
        if temp_constraint is not None:
            for entry in temp_constraint:
                atoms = [int(i) for i in entry[1].split()]
                current = float(entry[2])
                target = float(entry[3])
                if entry[0] == 'Distance(Angs)':
                    if len(atoms) == 2:
                        self.data['scan_constraint_sets']['stre'].append({'atoms': atoms, 'current': current, 'target': target})
                elif entry[0] == 'Angle':
                    if len(atoms) == 3:
                        self.data['scan_constraint_sets']['bend'].append({'atoms': atoms, 'current': current, 'target': target})
                elif entry[0] == 'Dihedral' and len(atoms) == 4:
                    self.data['scan_constraint_sets']['tors'].append({'atoms': atoms, 'current': current, 'target': target})

    def _read_pcm_information(self):
        """Parses information from PCM solvent calculations."""
        temp_dict = read_pattern(self.text, {'g_electrostatic': '\\s*G_electrostatic\\s+=\\s+([\\d\\-\\.]+)\\s+hartree\\s+=\\s+([\\d\\-\\.]+)\\s+kcal/mol\\s*', 'g_cavitation': '\\s*G_cavitation\\s+=\\s+([\\d\\-\\.]+)\\s+hartree\\s+=\\s+([\\d\\-\\.]+)\\s+kcal/mol\\s*', 'g_dispersion': '\\s*G_dispersion\\s+=\\s+([\\d\\-\\.]+)\\s+hartree\\s+=\\s+([\\d\\-\\.]+)\\s+kcal/mol\\s*', 'g_repulsion': '\\s*G_repulsion\\s+=\\s+([\\d\\-\\.]+)\\s+hartree\\s+=\\s+([\\d\\-\\.]+)\\s+kcal/mol\\s*', 'total_contribution_pcm': '\\s*Total\\s+=\\s+([\\d\\-\\.]+)\\s+hartree\\s+=\\s+([\\d\\-\\.]+)\\s+kcal/mol\\s*', 'solute_internal_energy': 'Solute Internal Energy \\(H0\\)\\s*=\\s*([\\d\\-\\.]+)'})
        for key in temp_dict:
            if temp_dict.get(key) is None:
                self.data['solvent_data'][key] = None
            elif len(temp_dict.get(key)) == 1:
                self.data['solvent_data'][key] = float(temp_dict.get(key)[0][0])
            else:
                temp_result = np.zeros(len(temp_dict.get(key)))
                for ii, entry in enumerate(temp_dict.get(key)):
                    temp_result[ii] = float(entry[0])
                self.data['solvent_data'][key] = temp_result

    def _read_smd_information(self):
        """Parses information from SMD solvent calculations."""
        temp_dict = read_pattern(self.text, {'smd0': 'E-EN\\(g\\) gas\\-phase elect\\-nuc energy\\s*([\\d\\-\\.]+) a\\.u\\.', 'smd3': 'G\\-ENP\\(liq\\) elect\\-nuc\\-pol free energy of system\\s*([\\d\\-\\.]+) a\\.u\\.', 'smd4': 'G\\-CDS\\(liq\\) cavity\\-dispersion\\-solvent structure\\s*free energy\\s*([\\d\\-\\.]+) kcal\\/mol', 'smd6': 'G\\-S\\(liq\\) free energy of system\\s*([\\d\\-\\.]+) a\\.u\\.', 'smd9': 'DeltaG\\-S\\(liq\\) free energy of\\s*solvation\\s*\\(9\\) = \\(6\\) \\- \\(0\\)\\s*([\\d\\-\\.]+) kcal\\/mol'})
        for key in temp_dict:
            if temp_dict.get(key) is None:
                self.data['solvent_data'][key] = None
            elif len(temp_dict.get(key)) == 1:
                self.data['solvent_data'][key] = float(temp_dict.get(key)[0][0])
            else:
                temp_result = np.zeros(len(temp_dict.get(key)))
                for ii, entry in enumerate(temp_dict.get(key)):
                    temp_result[ii] = float(entry[0])
                self.data['solvent_data'][key] = temp_result

    def _read_isosvp_information(self):
        """
        Parses information from ISOSVP solvent calculations.

        There are 5 energies output, as in the example below

        --------------------------------------------------------------------------------
        The Final SS(V)PE energies and Properties
        --------------------------------------------------------------------------------

        Energies
        --------------------
        The Final Solution-Phase Energy =     -40.4850599390
        The Solute Internal Energy =          -40.4846329759
        The Change in Solute Internal Energy =  0.0000121970  (   0.00765 KCAL/MOL)
        The Reaction Field Free Energy =       -0.0004269631  (  -0.26792 KCAL/MOL)
        The Total Solvation Free Energy =      -0.0004147661  (  -0.26027 KCAL/MOL)

        In addition, we need to parse the DIELST fortran variable to get the dielectric
        constant used.
        """
        temp_dict = read_pattern(self.text, {'final_soln_phase_e': '\\s*The Final Solution-Phase Energy\\s+=\\s+([\\d\\-\\.]+)\\s*', 'solute_internal_e': '\\s*The Solute Internal Energy\\s+=\\s+([\\d\\-\\.]+)\\s*', 'total_solvation_free_e': '\\s*The Total Solvation Free Energy\\s+=\\s+([\\d\\-\\.]+)\\s*', 'change_solute_internal_e': '\\s*The Change in Solute Internal Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'reaction_field_free_e': '\\s*The Reaction Field Free Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'isosvp_dielectric': '\\s*DIELST=\\s+(\\s+[\\d\\-\\.]+)\\s*'})
        for key in temp_dict:
            if temp_dict.get(key) is None:
                self.data['solvent_data']['isosvp'][key] = None
            elif len(temp_dict.get(key)) == 1:
                self.data['solvent_data']['isosvp'][key] = float(temp_dict.get(key)[0][0])
            else:
                temp_result = np.zeros(len(temp_dict.get(key)))
                for ii, entry in enumerate(temp_dict.get(key)):
                    temp_result[ii] = float(entry[0])
                self.data['solvent_data']['isosvp'][key] = temp_result

    def _read_cmirs_information(self):
        """
        Parses information from CMIRS solvent calculations.

        In addition to the 5 energies returned by ISOSVP (and read separately in
        _read_isosvp_information), there are 4 additional energies reported, as shown
        in the example below

        --------------------------------------------------------------------------------
        The Final SS(V)PE energies and Properties
        --------------------------------------------------------------------------------

        Energies
        --------------------
        The Final Solution-Phase Energy =     -40.4751881546
        The Solute Internal Energy =          -40.4748568841
        The Change in Solute Internal Energy =  0.0000089729  (   0.00563 KCAL/MOL)
        The Reaction Field Free Energy =       -0.0003312705  (  -0.20788 KCAL/MOL)
        The Dispersion Energy =                 0.6955550107  (  -2.27836 KCAL/MOL)
        The Exchange Energy =                   0.2652679507  (   2.15397 KCAL/MOL)
        Min. Negative Field Energy =            0.0005235850  (   0.00000 KCAL/MOL)
        Max. Positive Field Energy =            0.0179866718  (   0.00000 KCAL/MOL)
        The Total Solvation Free Energy =      -0.0005205275  (  -0.32664 KCAL/MOL)
        """
        temp_dict = read_pattern(self.text, {'dispersion_e': '\\s*The Dispersion Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'exchange_e': '\\s*The Exchange Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'min_neg_field_e': '\\s*Min. Negative Field Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*', 'max_pos_field_e': '\\s*Max. Positive Field Energy\\s+=\\s+(\\s+[\\d\\-\\.]+)\\s+\\(\\s+([\\d\\-\\.]+)\\s+KCAL/MOL\\)\\s*'})
        for key in temp_dict:
            if temp_dict.get(key) is None:
                self.data['solvent_data']['cmirs'][key] = None
            elif len(temp_dict.get(key)) == 1:
                self.data['solvent_data']['cmirs'][key] = float(temp_dict.get(key)[0][0])
            else:
                temp_result = np.zeros(len(temp_dict.get(key)))
                for ii, entry in enumerate(temp_dict.get(key)):
                    temp_result[ii] = float(entry[0])
                self.data['solvent_data']['cmirs'][key] = temp_result

    def _read_nbo_data(self):
        """Parses NBO output."""
        dfs = nbo_parser(self.filename)
        nbo_data = {}
        for key, value in dfs.items():
            nbo_data[key] = [df.to_dict() for df in value]
        self.data['nbo_data'] = nbo_data

    def _read_cdft(self):
        """Parses output from charge- or spin-constrained DFT (CDFT) calculations."""
        temp_dict = read_pattern(self.text, {'constraint': 'Constraint\\s+(\\d+)\\s+:\\s+([\\-\\.0-9]+)', 'multiplier': '\\s*Lam\\s+([\\.\\-0-9]+)'})
        self.data['cdft_constraints_multipliers'] = []
        for const, multip in zip(temp_dict.get('constraint', []), temp_dict.get('multiplier', [])):
            entry = {'index': int(const[0]), 'constraint': float(const[1]), 'multiplier': float(multip[0])}
            self.data['cdft_constraints_multipliers'].append(entry)
        header_pattern = '\\s*CDFT Becke Populations\\s*\\n\\-+\\s*\\n\\s*Atom\\s+Excess Electrons\\s+Population \\(a\\.u\\.\\)\\s+Net Spin'
        table_pattern = '\\s*(?:[0-9]+)\\s+(?:[A-Za-z0-9]+)\\s+([\\-\\.0-9]+)\\s+([\\.0-9]+)\\s+([\\-\\.0-9]+)'
        footer_pattern = '\\s*\\-+'
        becke_table = read_table_pattern(self.text, header_pattern, table_pattern, footer_pattern)
        if becke_table is None or len(becke_table) == 0:
            self.data['cdft_becke_excess_electrons'] = self.data['cdft_becke_population'] = self.data['cdft_becke_net_spin'] = None
        else:
            self.data['cdft_becke_excess_electrons'] = []
            self.data['cdft_becke_population'] = []
            self.data['cdft_becke_net_spin'] = []
            for table in becke_table:
                excess = []
                population = []
                spin = []
                for row in table:
                    excess.append(float(row[0]))
                    population.append(float(row[1]))
                    spin.append(float(row[2]))
                self.data['cdft_becke_excess_electrons'].append(excess)
                self.data['cdft_becke_population'].append(population)
                self.data['cdft_becke_net_spin'].append(spin)

    def _read_almo_msdft(self):
        """Parse output of ALMO(MSDFT) calculations for coupling between diabatic states."""
        temp_dict = read_pattern(self.text, {'states': 'Number of diabatic states: 2\\s*\\nstate 1\\s*\\ncharge per fragment\\s*\\n((?:\\s*[\\-0-9]+\\s*\\n)+)multiplicity per fragment\\s*\\n((?:\\s*[\\-0-9]+\\s*\\n)+)state 2\\s*\\ncharge per fragment\\s*\\n((?:\\s*[\\-0-9]+\\s*\\n)+)multiplicity per fragment\\s*\\n((?:\\s*[\\-0-9]+\\s*\\n)+)', 'diabat_energies': 'Energies of the diabats:\\s*\\n\\s*state 1:\\s+([\\-\\.0-9]+)\\s*\\n\\s*state 2:\\s+([\\-\\.0-9]+)', 'adiabat_energies': 'Energy of the adiabatic states\\s*\\n\\s*State 1:\\s+([\\-\\.0-9]+)\\s*\\n\\s*State 2:\\s+([\\-\\.0-9]+)', 'hamiltonian': 'Hamiltonian\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 'overlap': 'overlap\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 's2': '<S2>\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 'diabat_basis_coeff': 'Diabatic basis coefficients\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 'h_coupling': 'H passed to diabatic coupling calculation\\s*\\n\\s*1\\s+2\\s*\\n\\s*1\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)\\s*\\n\\s*2\\s+([\\-\\.0-9]+)\\s+([\\-\\.0-9]+)', 'coupling': 'Coupling between diabats 1 and 2: (?:[\\-\\.0-9]+) \\(([\\-\\.0-9]+) meV\\)'})
        if temp_dict.get('states') is None or len(temp_dict.get('states', [])) == 0:
            self.data['almo_coupling_states'] = None
        else:
            charges_1 = [int(r.strip()) for r in temp_dict['states'][0][0].strip().split('\n')]
            spins_1 = [int(r.strip()) for r in temp_dict['states'][0][1].strip().split('\n')]
            charges_2 = [int(r.strip()) for r in temp_dict['states'][0][2].strip().split('\n')]
            spins_2 = [int(r.strip()) for r in temp_dict['states'][0][3].strip().split('\n')]
            self.data['almo_coupling_states'] = [[[i, j] for i, j in zip(charges_1, spins_1)], [[i, j] for i, j in zip(charges_2, spins_2)]]
        if temp_dict.get('diabat_energies') is None or len(temp_dict.get('diabat_energies', [])) == 0:
            self.data['almo_diabat_energies_Hartree'] = None
        else:
            self.data['almo_diabat_energies_Hartree'] = [float(x) for x in temp_dict['diabat_energies'][0]]
        if temp_dict.get('adiabat_energies') is None or len(temp_dict.get('adiabat_energies', [])) == 0:
            self.data['almo_adiabat_energies_Hartree'] = None
        else:
            self.data['almo_adiabat_energies_Hartree'] = [float(x) for x in temp_dict['adiabat_energies'][0]]
        if temp_dict.get('hamiltonian') is None or len(temp_dict.get('hamiltonian', [])) == 0:
            self.data['almo_hamiltonian'] = None
        else:
            self.data['almo_hamiltonian'] = [[float(temp_dict['hamiltonian'][0][0]), float(temp_dict['hamiltonian'][0][1])], [float(temp_dict['hamiltonian'][0][2]), float(temp_dict['hamiltonian'][0][3])]]
        if temp_dict.get('overlap') is None or len(temp_dict.get('overlap', [])) == 0:
            self.data['almo_overlap_matrix'] = None
        else:
            self.data['almo_overlap_matrix'] = [[float(temp_dict['overlap'][0][0]), float(temp_dict['overlap'][0][1])], [float(temp_dict['overlap'][0][2]), float(temp_dict['overlap'][0][3])]]
        if temp_dict.get('s2') is None or len(temp_dict.get('s2', [])) == 0:
            self.data['almo_s2_matrix'] = None
        else:
            self.data['almo_s2_matrix'] = [[float(temp_dict['s2'][0][0]), float(temp_dict['s2'][0][1])], [float(temp_dict['s2'][0][2]), float(temp_dict['s2'][0][3])]]
        if temp_dict.get('diabat_basis_coeff') is None or len(temp_dict.get('diabat_basis_coeff', [])) == 0:
            self.data['almo_diabat_basis_coeff'] = None
        else:
            self.data['almo_diabat_basis_coeff'] = [[float(temp_dict['diabat_basis_coeff'][0][0]), float(temp_dict['diabat_basis_coeff'][0][1])], [float(temp_dict['diabat_basis_coeff'][0][2]), float(temp_dict['diabat_basis_coeff'][0][3])]]
        if temp_dict.get('h_coupling') is None or len(temp_dict.get('h_coupling', [])) == 0:
            self.data['almo_h_coupling_matrix'] = None
        else:
            self.data['almo_h_coupling_matrix'] = [[float(temp_dict['h_coupling'][0][0]), float(temp_dict['h_coupling'][0][1])], [float(temp_dict['h_coupling'][0][2]), float(temp_dict['h_coupling'][0][3])]]
        if temp_dict.get('coupling') is None or len(temp_dict.get('coupling', [])) == 0:
            self.data['almo_coupling_eV'] = None
        else:
            self.data['almo_coupling_eV'] = float(temp_dict['coupling'][0][0]) / 1000

    def _check_completion_errors(self):
        """Parses potential errors that can cause jobs to crash."""
        if read_pattern(self.text, {'key': 'Coordinates do not transform within specified threshold'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['failed_to_transform_coords']
        elif read_pattern(self.text, {'key': 'The Q\\-Chem input file has failed to pass inspection'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['input_file_error']
        elif read_pattern(self.text, {'key': 'Error opening input stream'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['failed_to_read_input']
        elif read_pattern(self.text, {'key': 'FileMan error: End of file reached prematurely'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['premature_end_FileMan_error']
        elif read_pattern(self.text, {'key': 'need to increase the array of NLebdevPts'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['NLebdevPts']
        elif read_pattern(self.text, {'key': 'method not available'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['method_not_available']
        elif read_pattern(self.text, {'key': 'Could not find \\$molecule section in ParseQInput'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['read_molecule_error']
        elif read_pattern(self.text, {'key': 'Welcome to Q-Chem'}, terminate_on_match=True).get('key') != [[]]:
            self.data['errors'] += ['never_called_qchem']
        elif read_pattern(self.text, {'key': '\\*\\*\\*ERROR\\*\\*\\* Hessian Appears to have all zero or negative eigenvalues'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['hessian_eigenvalue_error']
        elif read_pattern(self.text, {'key': 'FlexNet Licensing error'}, terminate_on_match=True).get('key') == [[]] or read_pattern(self.text, {'key': 'Unable to validate license'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['licensing_error']
        elif read_pattern(self.text, {'key': 'Could not open driver file in ReadDriverFromDisk'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['driver_error']
        elif read_pattern(self.text, {'key': 'Basis not supported for the above atom'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['basis_not_supported']
        elif read_pattern(self.text, {'key': 'Unable to find relaxed density'}, terminate_on_match=True).get('key') == [[]] or read_pattern(self.text, {'key': 'Out of Iterations- IterZ'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['failed_cpscf']
        elif read_pattern(self.text, {'key': 'RUN_NBO6 \\(rem variable\\) is not correct'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['bad_old_nbo6_rem']
        elif read_pattern(self.text, {'key': 'NBO_EXTERNAL \\(rem variable\\) is not correct'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['bad_new_nbo_external_rem']
        elif read_pattern(self.text, {'key': 'gen_scfman_exception:  GDM:: Zero or negative preconditioner scaling factor'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['gdm_neg_precon_error']
        elif read_pattern(self.text, {'key': 'too many atoms in ESPChgFit'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['esp_chg_fit_error']
        elif read_pattern(self.text, {'key': 'Please use larger MEM_STATIC'}, terminate_on_match=True).get('key') == [[]] or read_pattern(self.text, {'key': 'Please increase MEM_STATIC'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['mem_static_too_small']
        elif read_pattern(self.text, {'key': 'Please increase MEM_TOTAL'}, terminate_on_match=True).get('key') == [[]]:
            self.data['errors'] += ['mem_total_too_small']
        elif self.text[-34:-2] == 'Computing fast CPCM-SWIG hessian' or self.text[-16:-1] == 'Roots Converged':
            self.data['errors'] += ['probably_out_of_memory']
        else:
            tmp_failed_line_searches = read_pattern(self.text, {'key': '\\d+\\s+failed line searches\\.\\s+Resetting'}, terminate_on_match=False).get('key')
            if tmp_failed_line_searches is not None and len(tmp_failed_line_searches) > 10:
                self.data['errors'] += ['SCF_failed_to_converge']
        if self.data.get('errors') == []:
            self.data['errors'] += ['unknown_error']

    def as_dict(self):
        """
        Returns:
            MSONable dict.
        """
        dct = {}
        dct['data'] = self.data
        dct['text'] = self.text
        dct['filename'] = self.filename
        return jsanitize(dct, strict=True)