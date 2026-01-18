import numpy as np
import pytest
from ase.vibrations import Vibrations
from ase.calculators.h2morse import (H2Morse, H2MorseCalculator,
from ase.calculators.h2morse import (H2MorseExcitedStatesCalculator,
Check that traditional calling works