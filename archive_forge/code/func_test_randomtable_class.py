import random as pyrandom
import time
from functools import partial
from petl.util.random import randomseed, randomtable, RandomTable, dummytable, DummyTable
def test_randomtable_class():
    """
    Ensure that RandomTable provides a table with the right number of rows and columns.
    """
    columns, rows = (4, 60)
    table = RandomTable(numflds=columns, numrows=rows)
    assert len(table[0]) == columns
    assert len(table) == rows + 1