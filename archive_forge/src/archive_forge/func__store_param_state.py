import os
import sys
import re
import numpy as np
import subprocess
from contextlib import contextmanager
from pathlib import Path
from warnings import warn
from typing import Dict, Any
from xml.etree import ElementTree
import ase
from ase.io import read, jsonio
from ase.utils import PurePath
from ase.calculators import calculator
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.vasp.create_input import GenerateVaspInput
def _store_param_state(self):
    """Store current parameter state"""
    self.param_state = dict(float_params=self.float_params.copy(), exp_params=self.exp_params.copy(), string_params=self.string_params.copy(), int_params=self.int_params.copy(), input_params=self.input_params.copy(), bool_params=self.bool_params.copy(), list_int_params=self.list_int_params.copy(), list_bool_params=self.list_bool_params.copy(), list_float_params=self.list_float_params.copy(), dict_params=self.dict_params.copy())