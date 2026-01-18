import warnings
from chempy.units import allclose
from ..water_permittivity_bradley_pitzer_1979 import water_permittivity
from chempy.util.testing import requires
from chempy.units import linspace, units_library, default_units as u
@requires(units_library)
def test_water_permittivity__units():
    assert allclose(water_permittivity(298.15 * u.K, 1 * u.bar, units=u), 78.38436874203077)
    assert allclose(water_permittivity(linspace(297.5, 298.65) * u.K, 1 * u.bar, units=u), 78, rtol=0.01, atol=0.01)