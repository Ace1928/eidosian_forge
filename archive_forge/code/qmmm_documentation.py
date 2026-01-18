import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell

        A static method to import the the mapping from an estxyz file saved by
        export_extxyz() function
        Parameters
        ----------
        filename: string
            filename with saved configuration

        qm_calc: Calculator object
            QM-calculator.
        mm_calc: Calculator object
            MM-calculator (should be scaled, see :class:`RescaledCalculator`)
            Can use `ForceConstantCalculator` based on QM force constants, if
            available.

        Returns
        -------
        New object of ForceQMMM calculator with qm_selection_mask and
        qm_buffer_mask set according to the region array in the saved file
        