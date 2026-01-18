import os
import copy
from collections.abc import Iterable
from shutil import which
from typing import Dict, Optional
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError
removes list of keywords (delete) from kwargs