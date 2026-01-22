from __future__ import annotations
import os
import subprocess
import warnings
from glob import glob
from shutil import which
from typing import TYPE_CHECKING
import numpy as np
from monty.tempfile import ScratchDir
from pymatgen.core import Element
from pymatgen.io.vasp.inputs import Potcar
from pymatgen.io.vasp.outputs import Chgcar
def _from_data_dir(self, chargemol_output_path=None):
    """Internal command to parse Chargemol files from a directory.

        Args:
            chargemol_output_path (str): Path to the folder containing the
            Chargemol output files.
                Default: None (current working directory).
        """
    if chargemol_output_path is None:
        chargemol_output_path = '.'
    charge_path = f'{chargemol_output_path}/DDEC6_even_tempered_net_atomic_charges.xyz'
    self.ddec_charges = self._get_data_from_xyz(charge_path)
    self.dipoles = self._get_dipole_info(charge_path)
    bond_order_path = f'{chargemol_output_path}/DDEC6_even_tempered_bond_orders.xyz'
    if os.path.isfile(bond_order_path):
        self.bond_order_sums = self._get_data_from_xyz(bond_order_path)
        self.bond_order_dict = self._get_bond_order_info(bond_order_path)
    else:
        self.bond_order_sums = self.bond_order_dict = None
    spin_moment_path = f'{chargemol_output_path}/DDEC6_even_tempered_atomic_spin_moments.xyz'
    if os.path.isfile(spin_moment_path):
        self.ddec_spin_moments = self._get_data_from_xyz(spin_moment_path)
    else:
        self.ddec_spin_moments = None
    rsquared_path = f'{chargemol_output_path}/DDEC_atomic_Rsquared_moments.xyz'
    if os.path.isfile(rsquared_path):
        self.ddec_rsquared_moments = self._get_data_from_xyz(rsquared_path)
    else:
        self.ddec_rsquared_moments = None
    rcubed_path = f'{chargemol_output_path}/DDEC_atomic_Rcubed_moments.xyz'
    if os.path.isfile(rcubed_path):
        self.ddec_rcubed_moments = self._get_data_from_xyz(rcubed_path)
    else:
        self.ddec_rcubed_moments = None
    rfourth_path = f'{chargemol_output_path}/DDEC_atomic_Rfourth_moments.xyz'
    if os.path.isfile(rfourth_path):
        self.ddec_rfourth_moments = self._get_data_from_xyz(rfourth_path)
    else:
        self.ddec_rfourth_moments = None
    ddec_analysis_path = f'{chargemol_output_path}/VASP_DDEC_analysis.output'
    if os.path.isfile(ddec_analysis_path):
        self.cm5_charges = self._get_cm5_data_from_output(ddec_analysis_path)
    else:
        self.cm5_charges = None