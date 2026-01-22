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
def assertIdempotent(self, s):
    """Assert that quoting an unquoted string is a no-op and vice-versa.

        What matters here is that option values, as they appear in a store, can
        be safely round-tripped out of the store and back.

        :param s: A string, quoted if required.
        """
    self.assertEqual(s, self.store.quote(self.store.unquote(s)))
    self.assertEqual(s, self.store.unquote(self.store.quote(s)))