import os
import sys
import threading
from io import BytesIO
from textwrap import dedent
import configobj
from testtools import matchers
from .. import (bedding, branch, config, controldir, diff, errors, lock,
from .. import registry as _mod_registry
from .. import tests, trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..bzr import remote
from ..transport import remote as transport_remote
from . import features, scenarios, test_server
def test_triple_quotes(self):
    triple_quotes_value = 'spam\n""" that\'s my spam """\neggs'
    co = config.ConfigObj()
    co['test'] = triple_quotes_value
    outfile = BytesIO()
    co.write(outfile=outfile)
    output = outfile.getvalue()
    co2 = config.ConfigObj(BytesIO(output))
    self.assertEqual(triple_quotes_value, co2['test'])