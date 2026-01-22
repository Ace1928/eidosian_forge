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
@unittest.pytest.mark.default
@unittest.pytest.mark.neos
@unittest.skipIf(not neos_available, 'Cannot make connection to NEOS server')
@unittest.skipUnless(email_set, 'NEOS_EMAIL not set')
class TestKestrel(unittest.TestCase):

    def test_doc(self):
        kestrel = kestrelAMPL()
        tmp = [tuple(name.split(':')) for name in kestrel.solvers()]
        amplsolvers = set((v[0].lower() for v in tmp if v[1] == 'AMPL'))
        doc = pyomo.neos.doc
        dockeys = set(doc.keys())
        self.assertEqual(amplsolvers, dockeys)

    def test_connection_failed(self):
        try:
            orig_host = pyomo.neos.kestrel.NEOS.host
            pyomo.neos.kestrel.NEOS.host = 'neos-bogus-server.org'
            with LoggingIntercept() as LOG:
                kestrel = kestrelAMPL()
            self.assertIsNone(kestrel.neos)
            self.assertRegex(LOG.getvalue(), 'NEOS is temporarily unavailable:\\n\\t\\(.+\\)')
        finally:
            pyomo.neos.kestrel.NEOS.host = orig_host