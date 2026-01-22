import numpy as np
class Calculator:
    """ASE calculator.

    A calculator should store a copy of the atoms object used for the
    last calculation.  When one of the *get_potential_energy*,
    *get_forces*, or *get_stress* methods is called, the calculator
    should check if anything has changed since the last calculation
    and only do the calculation if it's really needed.  Two sets of
    atoms are considered identical if they have the same positions,
    atomic numbers, unit cell and periodic boundary conditions."""

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Return total energy.

        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        return 0.0

    def get_forces(self, atoms):
        """Return the forces."""
        return np.zeros((len(atoms), 3))

    def get_stress(self, atoms):
        """Return the stress."""
        return np.zeros(6)

    def calculation_required(self, atoms, quantities):
        """Check if a calculation is required.

        Check if the quantities in the *quantities* list have already
        been calculated for the atomic configuration *atoms*.  The
        quantities can be one or more of: 'energy', 'forces', 'stress',
        'charges' and 'magmoms'.

        This method is used to check if a quantity is available without
        further calculations.  For this reason, calculators should
        react to unknown/unsupported quantities by returning True,
        indicating that the quantity is *not* available."""
        return False