from __future__ import annotations
import warnings
import numpy as np
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.core import Element, Species
Utility method to convert an int (less than 20) to a roman numeral.