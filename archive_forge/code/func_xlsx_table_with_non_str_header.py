from __future__ import absolute_import, print_function, division
from datetime import datetime
from tempfile import NamedTemporaryFile
import pytest
import petl as etl
from petl.io.xlsx import fromxlsx, toxlsx, appendxlsx
from petl.test.helpers import ieq, eq_
@pytest.fixture(scope='module')
def xlsx_table_with_non_str_header():

    class Header:

        def __init__(self, name):
            self.__name = name

        def __str__(self):
            return self.__name

        def __eq__(self, other):
            return str(other) == str(self)
    return ((Header('foo'), Header('bar')), ('A', 1), ('B', 2), ('C', 2))