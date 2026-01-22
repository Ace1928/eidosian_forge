import os
import numpy as np
from copy import deepcopy
from ase.calculators.calculator import KPoints, kpts2kpts
def _format_brillouin_zone(array, name=None):
    out = ['  brillouin_zone']
    if name is not None:
        out += ['    zone_name {}'.format(name)]
    template = '    kvector' + ' {:20.16e}' * array.shape[1]
    for row in array:
        out.append(template.format(*row))
    out.append('  end')
    return out