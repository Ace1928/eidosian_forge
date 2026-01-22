from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
@unittest.skipIf(not xlsx_interface, 'No XLSX interface available')
class TestSpreadsheetXLSX(Spreadsheet, unittest.TestCase):
    _filename = 'excel.xlsx'