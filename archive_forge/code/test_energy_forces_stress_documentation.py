import numpy as np
from pytest import mark
from ase.lattice.cubic import FaceCenteredCubic

    To test that the calculator can produce correct energy and forces.  This
    is done by comparing the energy for an FCC argon lattice with an example
    model to the known value; the forces/stress returned by the model are
    compared to numerical estimates via finite difference.
    