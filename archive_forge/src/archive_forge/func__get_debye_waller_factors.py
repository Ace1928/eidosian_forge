import numpy as np
from ase.atoms import Atoms, symbols2numbers
from ase.utils import reader
from .utils import verify_cell_for_export, verify_dictionary
def _get_debye_waller_factors(self, DW):
    if np.isscalar(DW):
        if len(self.atom_types) > 1:
            raise ValueError('This cell contains more then one type of atoms and the Debye-Waller factor needs to be provided for each atom using a dictionary.')
        DW = np.ones_like(self.atoms.numbers) * DW
    elif isinstance(DW, dict):
        verify_dictionary(self.atoms, DW, 'DW')
        DW = {symbols2numbers(k)[0]: v for k, v in DW.items()}
        DW = np.vectorize(DW.get)(self.atoms.numbers)
    else:
        for name in ['DW', 'debye_waller_factors']:
            if name in self.atoms.arrays:
                DW = self.atoms.get_array(name)
    if DW is None:
        raise ValueError('Missing Debye-Waller factors. It can be provided as a dictionary with symbols as key or can be set for each atom by using the `set_array("debye_waller_factors", values)` of the `Atoms` object.')
    return DW