import pennylane as qml
from .hartree_fock import nuclear_energy, scf
from .observable_hf import fermionic_observable, qubit_observable
def _electron_integrals(*args):
    """Compute the one- and two-electron integrals in the molecular orbital basis.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            tuple[array[float]]: 1D tuple containing core constant, one- and two-electron integrals
        """
    _, coeffs, _, h_core, repulsion_tensor = scf(mol)(*args)
    one = qml.math.einsum('qr,rs,st->qt', coeffs.T, h_core, coeffs)
    two = qml.math.swapaxes(qml.math.einsum('ab,cd,bdeg,ef,gh->acfh', coeffs.T, coeffs.T, repulsion_tensor, coeffs, coeffs), 1, 3)
    core_constant = nuclear_energy(mol.nuclear_charges, mol.coordinates)(*args)
    if core is None and active is None:
        return (core_constant, one, two)
    for i in core:
        core_constant = core_constant + 2 * one[i][i]
        for j in core:
            core_constant = core_constant + 2 * two[i][j][j][i] - two[i][j][i][j]
    for p in active:
        for q in active:
            for i in core:
                o = qml.math.zeros(one.shape)
                o[p, q] = 1.0
                one = one + (2 * two[i][p][q][i] - two[i][p][i][q]) * o
    one = one[qml.math.ix_(active, active)]
    two = two[qml.math.ix_(active, active, active, active)]
    return (core_constant, one, two)