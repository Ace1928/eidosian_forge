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
class TestMutableStore(TestStore):
    scenarios = [(key, {'store_id': key, 'get_store': builder}) for key, builder in config.test_store_builder_registry.iteritems()]

    def setUp(self):
        super().setUp()
        self.transport = self.get_transport()

    def has_store(self, store):
        store_basename = urlutils.relative_url(self.transport.external_url(), store.external_url())
        return self.transport.has(store_basename)

    def test_save_empty_creates_no_file(self):
        if self.store_id in ('branch', 'remote_branch'):
            raise tests.TestNotApplicable('branch.conf is *always* created when a branch is initialized')
        store = self.get_store(self)
        store.save()
        self.assertEqual(False, self.has_store(store))

    def test_mutable_section_shared(self):
        store = self.get_store(self)
        store._load_from_string(b'foo=bar\n')
        if self.store_id in ('branch', 'remote_branch'):
            self.addCleanup(store.branch.lock_write().unlock)
        section1 = store.get_mutable_section(None)
        section2 = store.get_mutable_section(None)
        self.assertIs(section1, section2)

    def test_save_emptied_succeeds(self):
        store = self.get_store(self)
        store._load_from_string(b'foo=bar\n')
        if self.store_id in ('branch', 'remote_branch'):
            self.addCleanup(store.branch.lock_write().unlock)
        section = store.get_mutable_section(None)
        section.remove('foo')
        store.save()
        self.assertEqual(True, self.has_store(store))
        modified_store = self.get_store(self)
        sections = list(modified_store.get_sections())
        self.assertLength(0, sections)

    def test_save_with_content_succeeds(self):
        if self.store_id in ('branch', 'remote_branch'):
            raise tests.TestNotApplicable('branch.conf is *always* created when a branch is initialized')
        store = self.get_store(self)
        store._load_from_string(b'foo=bar\n')
        self.assertEqual(False, self.has_store(store))
        store.save()
        self.assertEqual(True, self.has_store(store))
        modified_store = self.get_store(self)
        sections = list(modified_store.get_sections())
        self.assertLength(1, sections)
        self.assertSectionContent((None, {'foo': 'bar'}), sections[0])

    def test_set_option_in_empty_store(self):
        store = self.get_store(self)
        if self.store_id in ('branch', 'remote_branch'):
            self.addCleanup(store.branch.lock_write().unlock)
        section = store.get_mutable_section(None)
        section.set('foo', 'bar')
        store.save()
        modified_store = self.get_store(self)
        sections = list(modified_store.get_sections())
        self.assertLength(1, sections)
        self.assertSectionContent((None, {'foo': 'bar'}), sections[0])

    def test_set_option_in_default_section(self):
        store = self.get_store(self)
        store._load_from_string(b'')
        if self.store_id in ('branch', 'remote_branch'):
            self.addCleanup(store.branch.lock_write().unlock)
        section = store.get_mutable_section(None)
        section.set('foo', 'bar')
        store.save()
        modified_store = self.get_store(self)
        sections = list(modified_store.get_sections())
        self.assertLength(1, sections)
        self.assertSectionContent((None, {'foo': 'bar'}), sections[0])

    def test_set_option_in_named_section(self):
        store = self.get_store(self)
        store._load_from_string(b'')
        if self.store_id in ('branch', 'remote_branch'):
            self.addCleanup(store.branch.lock_write().unlock)
        section = store.get_mutable_section('baz')
        section.set('foo', 'bar')
        store.save()
        modified_store = self.get_store(self)
        sections = list(modified_store.get_sections())
        self.assertLength(1, sections)
        self.assertSectionContent(('baz', {'foo': 'bar'}), sections[0])

    def test_load_hook(self):
        store = self.get_store(self)
        if self.store_id in ('branch', 'remote_branch'):
            self.addCleanup(store.branch.lock_write().unlock)
        section = store.get_mutable_section('baz')
        section.set('foo', 'bar')
        store.save()
        store = self.get_store(self)
        calls = []

        def hook(*args):
            calls.append(args)
        config.ConfigHooks.install_named_hook('load', hook, None)
        self.assertLength(0, calls)
        store.load()
        self.assertLength(1, calls)
        self.assertEqual((store,), calls[0])

    def test_save_hook(self):
        calls = []

        def hook(*args):
            calls.append(args)
        config.ConfigHooks.install_named_hook('save', hook, None)
        self.assertLength(0, calls)
        store = self.get_store(self)
        if self.store_id in ('branch', 'remote_branch'):
            self.addCleanup(store.branch.lock_write().unlock)
        section = store.get_mutable_section('baz')
        section.set('foo', 'bar')
        store.save()
        self.assertLength(1, calls)
        self.assertEqual((store,), calls[0])

    def test_set_mark_dirty(self):
        stack = config.MemoryStack(b'')
        self.assertLength(0, stack.store.dirty_sections)
        stack.set('foo', 'baz')
        self.assertLength(1, stack.store.dirty_sections)
        self.assertTrue(stack.store._need_saving())

    def test_remove_mark_dirty(self):
        stack = config.MemoryStack(b'foo=bar')
        self.assertLength(0, stack.store.dirty_sections)
        stack.remove('foo')
        self.assertLength(1, stack.store.dirty_sections)
        self.assertTrue(stack.store._need_saving())