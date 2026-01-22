import pytest
import itertools
from ase.cell import Cell
def all_pbcs():
    values = [False, True]
    yield from itertools.product(values, values, values)