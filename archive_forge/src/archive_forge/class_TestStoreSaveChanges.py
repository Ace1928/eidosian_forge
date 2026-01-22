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
class TestStoreSaveChanges(tests.TestCaseWithTransport):
    """Tests that config changes are kept in memory and saved on-demand."""

    def setUp(self):
        super().setUp()
        self.transport = self.get_transport()
        self.st1 = config.TransportIniFileStore(self.transport, 'foo.conf')
        self.st2 = config.TransportIniFileStore(self.transport, 'foo.conf')
        self.warnings = []

        def warning(*args):
            self.warnings.append(args[0] % args[1:])
        self.overrideAttr(trace, 'warning', warning)

    def has_store(self, store):
        store_basename = urlutils.relative_url(self.transport.external_url(), store.external_url())
        return self.transport.has(store_basename)

    def get_stack(self, store):
        return config.Stack([store.get_sections], store)

    def test_no_changes_no_save(self):
        s = self.get_stack(self.st1)
        s.store.save_changes()
        self.assertEqual(False, self.has_store(self.st1))

    def test_unrelated_concurrent_update(self):
        s1 = self.get_stack(self.st1)
        s2 = self.get_stack(self.st2)
        s1.set('foo', 'bar')
        s2.set('baz', 'quux')
        s1.store.save()
        self.assertEqual(None, s1.get('baz'))
        s2.store.save_changes()
        self.assertEqual('quux', s2.get('baz'))
        self.assertEqual('bar', s2.get('foo'))
        self.assertLength(0, self.warnings)

    def test_concurrent_update_modified(self):
        s1 = self.get_stack(self.st1)
        s2 = self.get_stack(self.st2)
        s1.set('foo', 'bar')
        s2.set('foo', 'baz')
        s1.store.save()
        s2.store.save_changes()
        self.assertEqual('baz', s2.get('foo'))
        self.assertLength(1, self.warnings)
        warning = self.warnings[0]
        self.assertStartsWith(warning, 'Option foo in section None')
        self.assertEndsWith(warning, 'was changed from <CREATED> to bar. The baz value will be saved.')

    def test_concurrent_deletion(self):
        self.st1._load_from_string(b'foo=bar')
        self.st1.save()
        s1 = self.get_stack(self.st1)
        s2 = self.get_stack(self.st2)
        s1.remove('foo')
        s2.remove('foo')
        s1.store.save_changes()
        self.assertLength(0, self.warnings)
        s2.store.save_changes()
        self.assertLength(1, self.warnings)
        warning = self.warnings[0]
        self.assertStartsWith(warning, 'Option foo in section None')
        self.assertEndsWith(warning, 'was changed from bar to <CREATED>. The <DELETED> value will be saved.')