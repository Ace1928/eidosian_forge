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
class TestIterOptionRefs(tests.TestCase):
    """iter_option_refs is a bit unusual, document some cases."""

    def assertRefs(self, expected, string):
        self.assertEqual(expected, list(config.iter_option_refs(string)))

    def test_empty(self):
        self.assertRefs([(False, '')], '')

    def test_no_refs(self):
        self.assertRefs([(False, 'foo bar')], 'foo bar')

    def test_single_ref(self):
        self.assertRefs([(False, ''), (True, '{foo}'), (False, '')], '{foo}')

    def test_broken_ref(self):
        self.assertRefs([(False, '{foo')], '{foo')

    def test_embedded_ref(self):
        self.assertRefs([(False, '{'), (True, '{foo}'), (False, '}')], '{{foo}}')

    def test_two_refs(self):
        self.assertRefs([(False, ''), (True, '{foo}'), (False, ''), (True, '{bar}'), (False, '')], '{foo}{bar}')

    def test_newline_in_refs_are_not_matched(self):
        self.assertRefs([(False, '{\nxx}{xx\n}{{\n}}')], '{\nxx}{xx\n}{{\n}}')