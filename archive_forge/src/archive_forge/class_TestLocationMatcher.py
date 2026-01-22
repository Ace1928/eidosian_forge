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
class TestLocationMatcher(TestStore):

    def setUp(self):
        super().setUp()
        self.get_store = config.test_store_builder_registry.get('configobj')

    def test_unrelated_section_excluded(self):
        store = self.get_store(self)
        store._load_from_string(b'\n[/foo]\nsection=/foo\n[/foo/baz]\nsection=/foo/baz\n[/foo/bar]\nsection=/foo/bar\n[/foo/bar/baz]\nsection=/foo/bar/baz\n[/quux/quux]\nsection=/quux/quux\n')
        self.assertEqual(['/foo', '/foo/baz', '/foo/bar', '/foo/bar/baz', '/quux/quux'], [section.id for _, section in store.get_sections()])
        matcher = config.LocationMatcher(store, '/foo/bar/quux')
        sections = [section for _, section in matcher.get_sections()]
        self.assertEqual(['/foo/bar', '/foo'], [section.id for section in sections])
        self.assertEqual(['quux', 'bar/quux'], [section.extra_path for section in sections])

    def test_more_specific_sections_first(self):
        store = self.get_store(self)
        store._load_from_string(b'\n[/foo]\nsection=/foo\n[/foo/bar]\nsection=/foo/bar\n')
        self.assertEqual(['/foo', '/foo/bar'], [section.id for _, section in store.get_sections()])
        matcher = config.LocationMatcher(store, '/foo/bar/baz')
        sections = [section for _, section in matcher.get_sections()]
        self.assertEqual(['/foo/bar', '/foo'], [section.id for section in sections])
        self.assertEqual(['baz', 'bar/baz'], [section.extra_path for section in sections])

    def test_appendpath_in_no_name_section(self):
        store = self.get_store(self)
        store._load_from_string(b'\nfoo=bar\nfoo:policy = appendpath\n')
        matcher = config.LocationMatcher(store, 'dir/subdir')
        sections = list(matcher.get_sections())
        self.assertLength(1, sections)
        self.assertEqual('bar/dir/subdir', sections[0][1].get('foo'))

    def test_file_urls_are_normalized(self):
        store = self.get_store(self)
        if sys.platform == 'win32':
            expected_url = 'file:///C:/dir/subdir'
            expected_location = 'C:/dir/subdir'
        else:
            expected_url = 'file:///dir/subdir'
            expected_location = '/dir/subdir'
        matcher = config.LocationMatcher(store, expected_url)
        self.assertEqual(expected_location, matcher.location)

    def test_branch_name_colo(self):
        store = self.get_store(self)
        store._load_from_string(dedent('            [/]\n            push_location=my{branchname}\n        ').encode('ascii'))
        matcher = config.LocationMatcher(store, 'file:///,branch=example%3c')
        self.assertEqual('example<', matcher.branch_name)
        (_, section), = matcher.get_sections()
        self.assertEqual('example<', section.locals['branchname'])

    def test_branch_name_basename(self):
        store = self.get_store(self)
        store._load_from_string(dedent('            [/]\n            push_location=my{branchname}\n        ').encode('ascii'))
        matcher = config.LocationMatcher(store, 'file:///parent/example%3c')
        self.assertEqual('example<', matcher.branch_name)
        (_, section), = matcher.get_sections()
        self.assertEqual('example<', section.locals['branchname'])