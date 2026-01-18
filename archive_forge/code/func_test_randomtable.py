import random as pyrandom
import time
from functools import partial
from petl.util.random import randomseed, randomtable, RandomTable, dummytable, DummyTable
def test_randomtable():
    """
    Ensure that randomtable provides a table with the right number of rows and columns.
    """
    columns, rows = (3, 10)
    table = randomtable(columns, rows)
    assert len(table[0]) == columns
    assert len(table) == rows + 1