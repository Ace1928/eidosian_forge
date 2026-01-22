import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
class RescaledCalculator(Calculator):
    """Rescales length and energy of a calculators to match given
    lattice constant and bulk modulus

    Useful for MM calculator used within a :class:`ForceQMMM` model.
    See T. D. Swinburne and J. R. Kermode, Phys. Rev. B 96, 144102 (2017)
    for a derivation of the scaling constants.
    """
    implemented_properties = ['forces', 'energy', 'stress']

    def __init__(self, mm_calc, qm_lattice_constant, qm_bulk_modulus, mm_lattice_constant, mm_bulk_modulus):
        Calculator.__init__(self)
        self.mm_calc = mm_calc
        self.alpha = qm_lattice_constant / mm_lattice_constant
        self.beta = mm_bulk_modulus / qm_bulk_modulus / self.alpha ** 3

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        scaled_atoms = atoms.copy()
        mm_cell = atoms.get_cell()
        scaled_atoms.set_cell(mm_cell / self.alpha, scale_atoms=True)
        results = {}
        if 'energy' in properties:
            energy = self.mm_calc.get_potential_energy(scaled_atoms)
            results['energy'] = energy / self.beta
        if 'forces' in properties:
            forces = self.mm_calc.get_forces(scaled_atoms)
            results['forces'] = forces / (self.beta * self.alpha)
        if 'stress' in properties:
            stress = self.mm_calc.get_stress(scaled_atoms)
            results['stress'] = stress / (self.beta * self.alpha ** 3)
        self.results = results