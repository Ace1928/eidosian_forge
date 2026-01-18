from ..nernst import nernst_potential
from chempy.util.testing import requires
from chempy.units import default_units, default_constants, units_library, allclose

    Test cases obtained from textbook examples of Nernst potential in cellular
    membranes. 310K = 37C, typical mammalian cell environment temperature.
    