import json
import logging
import time
from typing import Dict, Any
import numpy as np
from ase.calculators.calculator import Calculator, all_properties
Calculator wrapper to record and plot history of energy and function
    evaluations
    