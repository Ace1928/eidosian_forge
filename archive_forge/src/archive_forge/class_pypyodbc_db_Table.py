import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
@DataManagerFactory.register('pypyodbc', '%s database interface' % 'pypyodbc')
class pypyodbc_db_Table(pyodbc_db_Table):

    def __init__(self):
        pyodbc_db_Table.__init__(self)
        self.using = 'pypyodbc'

    def available(self):
        return pypyodbc_available

    def requirements(self):
        return 'pypyodbc'

    def connect(self, connection, options):
        assert options['using'] == 'pypyodbc'
        return pyodbc_db_Table.connect(self, connection, options)