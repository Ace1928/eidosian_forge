import os
import re
from pathlib import Path
from typing import Mapping
import configparser
import pytest
from ase.calculators.calculator import (names as calculator_names,
def _base_kw(self):
    from ase.units import Ry
    return dict(ecutwfc=300 / Ry)