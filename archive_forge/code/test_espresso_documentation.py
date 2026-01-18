import numpy as np
from ase import io
from ase import build
from ase.io.espresso import parse_position_line
from pytest import approx
Write a structure and read it back.