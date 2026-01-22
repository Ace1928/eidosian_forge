import collections
from math import sin, pi, sqrt
from numbers import Real, Integral
from typing import Any, Dict, Iterator, List, Sequence, Tuple, TypeVar, Union
import numpy as np
from ase.atoms import Atoms
import ase.units as units
import ase.io
from ase.utils import jsonable, lazymethod
from ase.calculators.singlepoint import SinglePointCalculator
from ase.spectrum.dosdata import RawDOSData
from ase.spectrum.doscollection import DOSCollection
@classmethod
def _tabulate_from_energies(cls, energies: Union[Sequence[complex], np.ndarray], im_tol: float=1e-08) -> List[str]:
    summary_lines = ['---------------------', '  #    meV     cm^-1', '---------------------']
    for n, e in enumerate(energies):
        if abs(e.imag) > im_tol:
            c = 'i'
            e = e.imag
        else:
            c = ''
            e = e.real
        summary_lines.append('{index:3d} {mev:6.1f}{im:1s}  {cm:7.1f}{im}'.format(index=n, mev=e * 1000.0, cm=e / units.invcm, im=c))
    summary_lines.append('---------------------')
    summary_lines.append('Zero-point energy: {:.3f} eV'.format(cls._calculate_zero_point_energy(energies=energies)))
    return summary_lines