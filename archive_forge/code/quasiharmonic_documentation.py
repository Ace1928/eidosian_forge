from __future__ import annotations
import logging
from collections import defaultdict
import numpy as np
from monty.dev import deprecated
from scipy.constants import physical_constants
from scipy.integrate import quadrature
from scipy.misc import derivative
from scipy.optimize import minimize
from pymatgen.analysis.eos import EOS, PolynomialEOS
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.due import Doi, due
Returns a dict with a summary of the computed properties.