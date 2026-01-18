import warnings
from chempy.util.testing import requires
from chempy.units import units_library
from ..water_diffusivity_holz_2000 import water_self_diffusion_coefficient as w_sd
def test_water_self_diffusion_coefficient():
    warnings.filterwarnings('error')
    assert abs(w_sd(273.15 + 0.0) - 1.099e-09) < 2.7e-11
    assert abs(w_sd(273.15 + 4.0) - 1.261e-09) < 1.1e-11
    assert abs(w_sd(273.15 + 10) - 1.525e-09) < 7e-12
    assert abs(w_sd(273.15 + 15) - 1.765e-09) < 6e-12
    assert abs(w_sd(273.15 + 20) - 2.023e-09) < 1e-12
    assert abs(w_sd(273.15 + 25) - 2.299e-09) < 1e-12
    assert abs(w_sd(273.15 + 30) - 2.594e-09) < 1e-12
    assert abs(w_sd(273.15 + 35) - 2.907e-09) < 4e-12
    try:
        w_sd(1)
    except UserWarning:
        pass
    else:
        raise
    warnings.resetwarnings()