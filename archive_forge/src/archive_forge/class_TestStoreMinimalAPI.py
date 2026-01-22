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
class TestStoreMinimalAPI(tests.TestCaseWithTransport):
    scenarios = [(key, {'get_store': builder}) for key, builder in config.test_store_builder_registry.iteritems()] + [('cmdline', {'get_store': lambda test: config.CommandLineStore()})]

    def test_id(self):
        store = self.get_store(self)
        if isinstance(store, config.TransportIniFileStore):
            raise tests.TestNotApplicable("%s is not a concrete Store implementation so it doesn't need an id" % (store.__class__.__name__,))
        self.assertIsNot(None, store.id)