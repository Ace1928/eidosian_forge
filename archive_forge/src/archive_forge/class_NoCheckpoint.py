from typing import Dict, Any
import numpy as np
import ase
from ase.db import connect
from ase.calculators.calculator import Calculator
class NoCheckpoint(Exception):
    pass