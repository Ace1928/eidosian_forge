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
def test_pathologically_hidden_list(self):
    self.conf.store._load_from_string(b'\nfoo=bin\nbar=go\nstart={foo\nmiddle=},{\nend=bar}\nhidden={start}{middle}{end}\n')
    self.registry.register(config.ListOption('hidden'))
    self.assertEqual(['bin', 'go'], self.conf.get('hidden', expand=True))