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
def test_registered_lazy_help(self):
    self.registry.register_lazy('lazy_foo', self.__module__, 'TestOptionRegistry.lazy_option')
    self.assertEqual('Lazy help', self.registry.get_help('lazy_foo'))