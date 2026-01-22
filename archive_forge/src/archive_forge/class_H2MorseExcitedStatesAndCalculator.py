from itertools import count
import numpy as np
from ase import Atoms
from ase.units import invcm, Ha
from ase.data import atomic_masses
from ase.calculators.calculator import all_changes
from ase.calculators.morse import MorsePotential
from ase.calculators.excitation_list import Excitation, ExcitationList
class H2MorseExcitedStatesAndCalculator(H2MorseExcitedStatesCalculator, H2MorseExcitedStates):
    """Traditional joined object for backward compatibility only"""

    def __init__(self, calculator, nstates=3):
        if isinstance(calculator, str):
            exlist = H2MorseExcitedStates.read(calculator, nstates)
        else:
            atoms = calculator.atoms
            atoms.calc = calculator
            excalc = H2MorseExcitedStatesCalculator(nstates)
            exlist = excalc.calculate(atoms)
        H2MorseExcitedStates.__init__(self, nstates=nstates)
        for ex in exlist:
            self.append(ex)