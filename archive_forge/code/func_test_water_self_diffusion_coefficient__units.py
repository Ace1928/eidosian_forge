import warnings
from chempy.util.testing import requires
from chempy.units import units_library
from ..water_diffusivity_holz_2000 import water_self_diffusion_coefficient as w_sd
@requires(units_library)
def test_water_self_diffusion_coefficient__units():
    from chempy.units import allclose, linspace, default_units as u
    unit = u.m ** 2 / u.s
    assert allclose(1000000000.0 * w_sd(298.15 * u.K, units=u), 2.299 * unit, rtol=0.001, atol=1e-08 * unit)
    assert allclose(1000000000.0 * w_sd(linspace(297, 299) * u.K, units=u), 2.299 * u.m ** 2 / u.s, rtol=0.05, atol=0.01 * unit)