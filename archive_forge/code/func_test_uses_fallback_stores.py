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
def test_uses_fallback_stores(self):
    self.overrideAttr(config, 'credential_store_registry', config.CredentialStoreRegistry())
    store = StubCredentialStore()
    store.add_credentials('http', 'example.com', 'joe', 'secret')
    config.credential_store_registry.register('stub', store, fallback=True)
    conf = config.AuthenticationConfig(_file=BytesIO())
    creds = conf.get_credentials('http', 'example.com')
    self.assertEqual('joe', creds['user'])
    self.assertEqual('secret', creds['password'])