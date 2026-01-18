import pytest
from pathlib import Path
from ase.parallel import parprint, world
from ase.vibrations.vibrations import Vibrations
from ase.vibrations.resonant_raman import ResonantRamanCalculator
from ase.vibrations.placzek import Placzek, Profeta
from ase.calculators.h2morse import (H2Morse,
Intensities of different Placzek implementations
    should be similar