import numpy as np
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.calculator import PropertyNotImplementedError
def arrays_to_kpoints(eigenvalues, occupations, weights):
    """Helper function for building SinglePointKPoints.

    Convert eigenvalue, occupation, and weight arrays to list of
    SinglePointKPoint objects."""
    nspins, nkpts, nbands = eigenvalues.shape
    assert eigenvalues.shape == occupations.shape
    assert len(weights) == nkpts
    kpts = []
    for s in range(nspins):
        for k in range(nkpts):
            kpt = SinglePointKPoint(weight=weights[k], s=s, k=k, eps_n=eigenvalues[s, k], f_n=occupations[s, k])
            kpts.append(kpt)
    return kpts