import os.path
import re
import sys
import shutil
from decimal import Decimal
from pyomo.common.dependencies import attempt_import
from pyomo.dataportal import TableData
from pyomo.dataportal.factory import DataManagerFactory
@DataManagerFactory.register('sqlite3', 'sqlite3 database interface')
class sqlite3_db_Table(db_Table):

    def __init__(self):
        db_Table.__init__(self)
        self.using = 'sqlite3'

    def available(self):
        return sqlite3_available

    def requirements(self):
        return 'sqlite3'

    def connect(self, connection, options):
        assert options['using'] == 'sqlite3'
        filename = connection
        if not os.path.exists(filename):
            raise Exception('No such file: ' + filename)
        con = sqlite3.connect(filename)
        if options.text_factory:
            con.text_factory = options.text_factory
        return con