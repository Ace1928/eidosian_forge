import os
import json
import os.path
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.scripting.pyomo_main import main
from pyomo.scripting.util import cleanup
from pyomo.neos.kestrel import kestrelAMPL
import pyomo.neos
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
def test_ooqp(self):
    if self.sense == pyo.maximize:
        with self.assertRaisesRegex(AssertionError, '.* != 1 within'):
            self._run('ooqp')
    else:
        self._run('ooqp')