import pytest
from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.calculators.h2morse import (H2Morse,
from ase.vibrations.albrecht import Albrecht
def test_one_state(testdir, rrname, atoms):
    om = 1
    gam = 0.1
    with Albrecht(atoms, H2MorseExcitedStates, exkwargs={'nstates': 1}, name=rrname, overlap=True, approximation='Albrecht A', txt=None) as ao:
        aoi = ao.get_absolute_intensities(omega=om, gamma=gam)[-1]
    with Albrecht(atoms, H2MorseExcitedStates, exkwargs={'nstates': 1}, name=rrname, approximation='Albrecht A', txt=None) as al:
        ali = al.get_absolute_intensities(omega=om, gamma=gam)[-1]
    assert ali == pytest.approx(aoi, 1e-09)