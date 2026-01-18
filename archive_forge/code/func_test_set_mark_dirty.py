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
def test_set_mark_dirty(self):
    stack = config.MemoryStack(b'')
    self.assertLength(0, stack.store.dirty_sections)
    stack.set('foo', 'baz')
    self.assertLength(1, stack.store.dirty_sections)
    self.assertTrue(stack.store._need_saving())