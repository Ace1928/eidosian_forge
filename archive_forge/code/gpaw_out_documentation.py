import re
from typing import List, Tuple, Union
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
Read text output from GPAW calculation.