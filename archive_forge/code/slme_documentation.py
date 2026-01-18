from __future__ import annotations
import os
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import constants
from scipy.constants import physical_constants, speed_of_light
from scipy.integrate import simps
from scipy.interpolate import interp1d
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.util.due import Doi, due

    Calculate the SLME.

    Args:
        material_energy_for_absorbance_data: energy grid for absorbance data
        material_absorbance_data: absorption coefficient in m^-1
        material_direct_allowed_gap: direct bandgap in eV
        material_indirect_gap: indirect bandgap in eV
        thickness: thickness of the material in m
        temperature: working temperature in K
        absorbance_in_inverse_centimeters: whether the absorbance data is in the unit of cm^-1
        cut_off_absorbance_below_direct_allowed_gap: whether to discard all absorption below bandgap
        plot_current_voltage: whether to plot the current-voltage curve

    Returns:
        The calculated maximum efficiency.
    