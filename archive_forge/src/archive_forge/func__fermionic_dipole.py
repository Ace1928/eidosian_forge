import pennylane as qml
from pennylane.fermi import FermiSentence, FermiWord
from .basis_data import atomic_numbers
from .hartree_fock import scf
from .matrices import moment_matrix
from .observable_hf import fermionic_observable, qubit_observable
def _fermionic_dipole(*args):
    """Build the fermionic dipole moment observable.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            FermiSentence: fermionic dipole moment
        """
    constants, integrals = dipole_integrals(mol, core, active)(*args)
    nd = [qml.math.array([0]), qml.math.array([0]), qml.math.array([0])]
    for i, s in enumerate(mol.symbols):
        nd[0] = nd[0] + atomic_numbers[s] * mol.coordinates[i][0]
        nd[1] = nd[1] + atomic_numbers[s] * mol.coordinates[i][1]
        nd[2] = nd[2] + atomic_numbers[s] * mol.coordinates[i][2]
    d_ferm = []
    for i in range(3):
        f = fermionic_observable(constants[i], integrals[i], cutoff=cutoff)
        d_ferm.append(FermiSentence({FermiWord({}): nd[i][0]}) - f)
    return d_ferm