from __future__ import annotations
import itertools
import os
import warnings
import numpy as np
from ruamel.yaml import YAML
from pymatgen.core import SETTINGS
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Element, Molecule, Structure
from pymatgen.io.cp2k.inputs import (
from pymatgen.io.cp2k.utils import get_truncated_coulomb_cutoff, get_unique_site_indices
from pymatgen.io.vasp.inputs import Kpoints as VaspKpoints
from pymatgen.io.vasp.inputs import KpointsSupportedModes
def activate_hybrid(self, hybrid_functional: str='PBE0', hf_fraction: float=0.25, gga_x_fraction: float=0.75, gga_c_fraction: float=1, max_memory: int=2000, cutoff_radius: float=8.0, potential_type: str | None=None, omega: float=0.11, scale_coulomb: float=1, scale_gaussian: float=1, scale_longrange: float=1, admm: bool=True, admm_method: str='BASIS_PROJECTION', admm_purification_method: str='NONE', admm_exch_correction_func: str='DEFAULT', eps_schwarz: float=1e-07, eps_schwarz_forces: float=1e-06, screen_on_initial_p: bool=True, screen_p_forces: bool=True) -> None:
    """
        Basic set for activating hybrid DFT calculation using Auxiliary Density Matrix Method.

        Note 1: When running ADMM with cp2k, memory is very important. If the memory requirements
        exceed what is available (see max_memory), then CP2K will have to calculate the 4-electron
        integrals for HFX during each step of the SCF cycle. ADMM provides a huge speed up by
        making the memory requirements *feasible* to fit into RAM, which means you only need to
        calculate the integrals once each SCF cycle. But, this only works if it fits into memory.
        When setting up ADMM calculations, we recommend doing whatever is possible to fit all the
        4EI into memory.

        Note 2: This set is designed for reliable high-throughput calculations, NOT for extreme
        accuracy. Please review the in-line comments in this method if you want more control.

        Args:
            hybrid_functional (str): Type of hybrid functional. This set supports HSE (screened)
                and PBE0 (truncated). Default is PBE0, which converges easier in the GPW basis
                used by cp2k.
            hf_fraction (float): fraction of exact HF exchange energy to mix. Default: 0.25
            gga_x_fraction (float): fraction of gga exchange energy to retain. Default: 0.75
            gga_c_fraction (float): fraction of gga correlation energy to retain. Default: 1.0
            max_memory (int): Maximum memory available to each MPI process (in Mb) in the
                calculation. Most modern computing nodes will have ~2Gb per core, or 2048 Mb,
                but check for your specific system. This value should be as large as possible
                while still leaving some memory for the other parts of cp2k. Important: If
                this value is set larger than the memory limits, CP2K will likely seg-fault.
                Default: 2000
            cutoff_radius (float): for truncated hybrid functional (i.e. PBE0), this is the cutoff
                radius. The default is selected as that which generally gives convergence, but
                maybe too low (if you want very high accuracy) or too high (if you want a quick
                screening). Default: 8 angstroms
            potential_type (str): what interaction potential to use for HFX. Available in CP2K are
                COULOMB, GAUSSIAN, IDENTITY, LOGRANGE, MIX_CL, MIX_CL_TRUNC, MIX_LG, SHORTRANGE,
                and TRUNCATED. Default is None, and it will be set automatically depending on the
                named hybrid_functional that you use, but setting it to one of the acceptable
                values will constitute a user-override.
            omega (float): For HSE, this specifies the screening parameter. HSE06 sets this as
                0.2, which is the default.
            scale_coulomb: Scale for the coulomb operator if using a range separated functional
            scale_gaussian: Scale for the gaussian operator (if applicable)
            scale_longrange: Scale for the coulomb operator if using a range separated functional
            admm: Whether or not to use the auxiliary density matrix method for the exact
                HF exchange contribution. Highly recommended. Speed ups between 10x and 1000x are
                possible when compared to non ADMM hybrid calculations.
            admm_method: Method for constructing the auxiliary basis
            admm_purification_method: Method for purifying the auxiliary density matrix so as to
                preserve properties, such as idempotency. May lead to shifts in the
                eigenvalues.
            admm_exch_correction_func: Which functional to use to calculate the exchange correction
                E_x(primary) - E_x(aux)
            eps_schwarz: Screening threshold for HFX, in Ha. Contributions smaller than
                this will be screened. The smaller the value, the more accurate, but also the more
                costly. Default value is 1e-7. 1e-6 works in a large number of cases, but is
                quite aggressive, which can lead to convergence issues.
            eps_schwarz_forces: Same as for eps_schwarz, but for screening contributions to
                forces. Convergence is not as sensitive with respect to eps_schwarz forces as
                compared to eps_schwarz, and so 1e-6 should be good default.
            screen_on_initial_p: If an initial density matrix is provided, in the form of a
                CP2K wfn restart file, then this initial density will be used for screening. This
                is generally very computationally efficient, but, as with eps_schwarz, can lead to
                instabilities if the initial density matrix is poor.
            screen_p_forces: Same as screen_on_initial_p, but for screening of forces.
        """
    if not admm:
        for k, v in self.basis_and_potential.items():
            if 'aux_basis' in v:
                del self.basis_and_potential[k]['aux_basis']
        del self['force_eval']['subsys']
        self.create_subsys(self.structure)
    else:
        aux_matrix_params = {'ADMM_PURIFICATION_METHOD': Keyword('ADMM_PURIFICATION_METHOD', admm_purification_method), 'METHOD': Keyword('METHOD', admm_method), 'EXCH_CORRECTION_FUNC': Keyword('EXCH_CORRECTION_FUNC', admm_exch_correction_func)}
        aux_matrix = Section('AUXILIARY_DENSITY_MATRIX_METHOD', keywords=aux_matrix_params, subsections={})
        self.subsections['FORCE_EVAL']['DFT'].insert(aux_matrix)
    screening = Section('SCREENING', subsections={}, keywords={'EPS_SCHWARZ': Keyword('EPS_SCHWARZ', eps_schwarz), 'EPS_SCHWARZ_FORCES': Keyword('EPS_SCHWARZ_FORCES', eps_schwarz_forces), 'SCREEN_ON_INITIAL_P': Keyword('SCREEN_ON_INITIAL_P', screen_on_initial_p), 'SCREEN_P_FORCES': Keyword('SCREEN_P_FORCES', screen_p_forces)})
    if isinstance(self.structure, Structure):
        max_cutoff_radius = get_truncated_coulomb_cutoff(self.structure)
        if max_cutoff_radius < cutoff_radius:
            warnings.warn("Provided cutoff radius exceeds half the minimum distance between atoms. I hope you know what you're doing.")
    ip_keywords = {}
    if hybrid_functional == 'HSE06':
        pbe = PBE('ORIG', scale_c=1, scale_x=0)
        xc_functional = XCFunctional(functionals=[], subsections={'PBE': pbe})
        potential_type = potential_type or 'SHORTRANGE'
        xc_functional.insert(Section('XWPBE', subsections={}, keywords={'SCALE_X0': Keyword('SCALE_X0', 1), 'SCALE_X': Keyword('SCALE_X', -0.25), 'OMEGA': Keyword('OMEGA', 0.11)}))
        ip_keywords.update({'POTENTIAL_TYPE': Keyword('POTENTIAL_TYPE', potential_type), 'OMEGA': Keyword('OMEGA', 0.11), 'CUTOFF_RADIUS': Keyword('CUTOFF_RADIUS', cutoff_radius)})
    elif hybrid_functional == 'PBE0':
        pbe = PBE('ORIG', scale_c=1, scale_x=1 - hf_fraction)
        xc_functional = XCFunctional(functionals=[], subsections={'PBE': pbe})
        if isinstance(self.structure, Molecule):
            potential_type = 'COULOMB'
        else:
            potential_type = 'TRUNCATED'
            xc_functional.insert(Section('PBE_HOLE_T_C_LR', subsections={}, keywords={'CUTOFF_RADIUS': Keyword('CUTOFF_RADIUS', cutoff_radius), 'SCALE_X': Keyword('SCALE_X', hf_fraction)}))
            ip_keywords['CUTOFF_RADIUS'] = Keyword('CUTOFF_RADIUS', cutoff_radius)
            ip_keywords['T_C_G_DATA'] = Keyword('T_C_G_DATA', 't_c_g.dat')
        ip_keywords['POTENTIAL_TYPE'] = Keyword('POTENTIAL_TYPE', potential_type)
    elif hybrid_functional == 'RSH':
        pbe = PBE('ORIG', scale_c=1, scale_x=0)
        xc_functional = XCFunctional(functionals=[], subsections={'PBE': pbe})
        potential_type = potential_type or 'MIX_CL_TRUNC'
        hf_fraction = 1
        ip_keywords.update({'POTENTIAL_TYPE': Keyword('POTENTIAL_TYPE', potential_type), 'CUTOFF_RADIUS': Keyword('CUTOFF_RADIUS', cutoff_radius), 'T_C_G_DATA': Keyword('T_C_G_DATA', 't_c_g.dat'), 'OMEGA': Keyword('OMEGA', omega), 'SCALE_COULOMB': Keyword('SCALE_COULOMB', scale_coulomb), 'SCALE_LONGRANGE': Keyword('SCALE_LONGRANGE', scale_longrange - scale_coulomb)})
        xc_functional.insert(Section('XWPBE', subsections={}, keywords={'SCALE_X0': Keyword('SCALE_X0', 1 - scale_longrange), 'SCALE_X': Keyword('SCALE_X', scale_longrange - scale_coulomb), 'OMEGA': Keyword('OMEGA', omega)}))
        xc_functional.insert(Section('PBE_HOLE_T_C_LR', subsections={}, keywords={'CUTOFF_RADIUS': Keyword('CUTOFF_RADIUS', cutoff_radius), 'SCALE_X': Keyword('SCALE_X', scale_longrange)}))
    else:
        warnings.warn('Unknown hybrid functional. Using PBE base functional and overriding all settings manually. Proceed with caution.')
        pbe = PBE('ORIG', scale_c=gga_c_fraction, scale_x=gga_x_fraction)
        xc_functional = XCFunctional(functionals=[], subsections={'PBE': pbe})
        ip_keywords.update({'POTENTIAL_TYPE': Keyword('POTENTIAL_TYPE', potential_type), 'CUTOFF_RADIUS': Keyword('CUTOFF_RADIUS', cutoff_radius), 'T_C_G_DATA': Keyword('T_C_G_DATA', 't_c_g.dat'), 'SCALE_COULOMB': Keyword('SCALE_COULOMB', scale_coulomb), 'SCALE_GAUSSIAN': Keyword('SCALE_GAUSSIAN', scale_gaussian), 'SCALE_LONGRANGE': Keyword('SCALE_LONGRANGE', scale_longrange), 'OMEGA': Keyword('OMEGA', omega)})
    interaction_potential = Section('INTERACTION_POTENTIAL', subsections={}, keywords=ip_keywords)
    load_balance = Section('LOAD_BALANCE', keywords={'RANDOMIZE': Keyword('RANDOMIZE', True)}, subsections={})
    memory = Section('MEMORY', subsections={}, keywords={'EPS_STORAGE_SCALING': Keyword('EPS_STORAGE_SCALING', 0.1), 'MAX_MEMORY': Keyword('MAX_MEMORY', max_memory)})
    hf = Section('HF', keywords={'FRACTION': Keyword('FRACTION', hf_fraction)}, subsections={'SCREENING': screening, 'INTERACTION_POTENTIAL': interaction_potential, 'LOAD_BALANCE': load_balance, 'MEMORY': memory})
    xc = Section('XC', subsections={'XC_FUNCTIONAL': xc_functional, 'HF': hf})
    self.subsections['FORCE_EVAL']['DFT'].insert(xc)