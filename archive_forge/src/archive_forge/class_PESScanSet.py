from __future__ import annotations
import logging
import os
import warnings
from typing import TYPE_CHECKING, Literal
from monty.io import zopen
from pymatgen.io.qchem.inputs import QCInput
from pymatgen.io.qchem.utils import lower_and_check_unique
class PESScanSet(QChemDictSet):
    """
    QChemDictSet for a potential energy surface scan (PES_SCAN) calculation,
    used primarily to identify possible transition states or to sample different
    geometries.
    Note: Because there are no defaults that can be used for a PES scan (the
    variables are completely dependent on the molecular structure), by default
    scan_variables = None. However, a PES Scan job should not be run with less
    than one variable (or more than two variables).
    """

    def __init__(self, molecule: Molecule, basis_set: str='def2-svpd', scf_algorithm: str='diis', qchem_version: int=5, dft_rung: int=4, pcm_dielectric: float | None=None, isosvp_dielectric: float | None=None, smd_solvent: str | None=None, cmirs_solvent: Literal['water', 'acetonitrile', 'dimethyl sulfoxide', 'cyclohexane', 'benzene'] | None=None, custom_smd: str | None=None, max_scf_cycles: int=100, plot_cubes: bool=False, nbo_params: dict | None=None, opt_variables: dict[str, list] | None=None, scan_variables: dict[str, list] | None=None, overwrite_inputs: dict | None=None, vdw_mode: Literal['atomic', 'sequential']='atomic'):
        """
        Args:
            molecule (Pymatgen Molecule object)
            opt_variables (dict): A dictionary of opt sections, where each opt section is a key
                and the corresponding values are a list of strings. Strings must be formatted
                as instructed by the QChem manual. The different opt sections are: CONSTRAINT, FIXED,
                DUMMY, and CONNECT.

                Ex. opt = {"CONSTRAINT": ["tors 2 3 4 5 25.0", "tors 2 5 7 9 80.0"], "FIXED": ["2 XY"]}
            scan_variables (dict): A dictionary of scan variables. Because two constraints of the
                same type are allowed (for instance, two torsions or two bond stretches), each TYPE of
                variable (stre, bend, tors) should be its own key in the dict, rather than each variable.
                Note that the total number of variable (sum of lengths of all lists) CANNOT be more than two.

                Ex. scan_variables = {"stre": ["3 6 1.5 1.9 0.1"], "tors": ["1 2 3 4 -180 180 15"]}
            basis_set (str): Basis set to use. (Default: "def2-svpd")
            scf_algorithm (str): Algorithm to use for converging the SCF. Recommended choices are
                "DIIS", "GDM", and "DIIS_GDM". Other algorithms supported by Qchem's GEN_SCFMAN
                module will also likely perform well. Refer to the QChem manual for further details.
                (Default: "diis")
            qchem_version (int): Which major version of Q-Chem will be run. Supports 5 and 6. (Default: 5)
            dft_rung (int): Select the rung on "Jacob's Ladder of Density Functional Approximations" in
                order of increasing accuracy/cost. For each rung, we have prescribed one functional based
                on our experience, available benchmarks, and the suggestions of the Q-Chem manual:
                1 (LSDA) = SPW92
                2 (GGA) = B97-D3(BJ)
                3 (metaGGA) = B97M-V
                4 (hybrid metaGGA) = ωB97M-V
                5 (double hybrid metaGGA) = ωB97M-(2)

                (Default: 4)

                To set a functional not given by one of the above, set the overwrite_inputs
                argument to {"method":"<NAME OF FUNCTIONAL>"}
            pcm_dielectric (float): Dielectric constant to use for PCM implicit solvation model. (Default: None)
                If supplied, will set up the $pcm section of the input file for a C-PCM calculation.
                Other types of PCM calculations (e.g., IEF-PCM, SS(V)PE, etc.) may be requested by passing
                custom keywords to overwrite_inputs, e.g.
                overwrite_inputs = {"pcm": {"theory": "ssvpe"}}
                Refer to the QChem manual for further details on the models available.

                **Note that only one of pcm_dielectric, isosvp_dielectric, smd_solvent, or cmirs_solvent may be set.**
            isosvp_dielectric (float): Dielectric constant to use for isodensity SS(V)PE implicit solvation model.
                (Default: None). If supplied, will set solvent_method to "isosvp" and populate the $svp section
                of the input file with appropriate parameters.

                **Note that only one of pcm_dielectric, isosvp_dielectric, smd_solvent, or cmirs_solvent may be set.**
            smd_solvent (str): Solvent to use for SMD implicit solvation model. (Default: None)
                Examples include "water", "ethanol", "methanol", and "acetonitrile". Refer to the QChem
                manual for a complete list of solvents available. To define a custom solvent, set this
                argument to "custom" and populate custom_smd with the necessary parameters.

                **Note that only one of pcm_dielectric, isosvp_dielectric, smd_solvent, or cmirs_solvent may be set.**
            cmirs_solvent (str): Solvent to use for the CMIRS implicit solvation model. (Default: None).
                Only 5 solvents are presently available as of Q-Chem 6: "water", "benzene", "cyclohexane",
                "dimethyl sulfoxide", and "acetonitrile". Note that selection of a solvent here will also
                populate the iso SS(V)PE dielectric constant, because CMIRS uses the isodensity SS(V)PE model
                to compute electrostatics.

                **Note that only one of pcm_dielectric, isosvp_dielectric, smd_solvent, or cmirs_solvent may be set.**
            custom_smd (str): List of parameters to define a custom solvent in SMD. (Default: None)
                Must be given as a string of seven comma separated values in the following order:
                "dielectric, refractive index, acidity, basicity, surface tension, aromaticity,
                electronegative halogenicity"
                Refer to the QChem manual for further details.
            max_scf_cycles (int): Maximum number of SCF iterations. (Default: 100)
            plot_cubes (bool): Whether to write CUBE files of the electron density. (Default: False)
            overwrite_inputs (dict): Dictionary of QChem input sections to add or overwrite variables.
                The currently available sections (keys) are rem, pcm,
                solvent, smx, opt, scan, van_der_waals, and plots. The value of each key is a
                dictionary of key value pairs relevant to that section. For example, to add
                a new variable to the rem section that sets symmetry to false, use

                overwrite_inputs = {"rem": {"symmetry": "false"}}

                **Note that if something like basis is added to the rem dict it will overwrite
                the default basis.**

                **Note that supplying a van_der_waals section here will automatically modify
                the PCM "radii" setting to "read".**

                **Note that all keys must be given as strings, even when they are numbers!**
            vdw_mode ('atomic' | 'sequential'): Method of specifying custom van der Waals radii. Applies only if
                you are using overwrite_inputs to add a $van_der_waals section to the input. In 'atomic' mode
                (default), dict keys represent the atomic number associated with each radius (e.g., '12' = carbon).
                In 'sequential' mode, dict keys represent the sequential position of a single
                specific atom in the input structure.
        """
        self.basis_set = basis_set
        self.scf_algorithm = scf_algorithm
        self.max_scf_cycles = max_scf_cycles
        if scan_variables is None:
            raise ValueError('Cannot run a pes_scan job without some variable to scan over!')
        super().__init__(molecule=molecule, job_type='pes_scan', dft_rung=dft_rung, pcm_dielectric=pcm_dielectric, isosvp_dielectric=isosvp_dielectric, smd_solvent=smd_solvent, cmirs_solvent=cmirs_solvent, custom_smd=custom_smd, opt_variables=opt_variables, scan_variables=scan_variables, basis_set=self.basis_set, scf_algorithm=self.scf_algorithm, qchem_version=qchem_version, max_scf_cycles=self.max_scf_cycles, plot_cubes=plot_cubes, nbo_params=nbo_params, overwrite_inputs=overwrite_inputs, vdw_mode=vdw_mode)