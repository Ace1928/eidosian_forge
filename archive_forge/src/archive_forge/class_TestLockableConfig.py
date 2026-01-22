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
class TestLockableConfig(tests.TestCaseInTempDir):
    scenarios = lockable_config_scenarios()
    config_class = None
    config_args = None
    config_section = None

    def setUp(self):
        super().setUp()
        self._content = '[{}]\none=1\ntwo=2\n'.format(self.config_section)
        self.config = self.create_config(self._content)

    def get_existing_config(self):
        return self.config_class(*self.config_args)

    def create_config(self, content):
        kwargs = dict(save=True)
        c = self.config_class.from_string(content, *self.config_args, **kwargs)
        return c

    def test_simple_read_access(self):
        self.assertEqual('1', self.config.get_user_option('one'))

    def test_simple_write_access(self):
        self.config.set_user_option('one', 'one')
        self.assertEqual('one', self.config.get_user_option('one'))

    def test_listen_to_the_last_speaker(self):
        c1 = self.config
        c2 = self.get_existing_config()
        c1.set_user_option('one', 'ONE')
        c2.set_user_option('two', 'TWO')
        self.assertEqual('ONE', c1.get_user_option('one'))
        self.assertEqual('TWO', c2.get_user_option('two'))
        self.assertEqual('ONE', c2.get_user_option('one'))

    def test_last_speaker_wins(self):
        c1 = self.config
        c2 = self.get_existing_config()
        c1.set_user_option('one', 'c1')
        c2.set_user_option('one', 'c2')
        self.assertEqual('c2', c2._get_user_option('one'))
        self.assertEqual('c1', c1._get_user_option('one'))
        c1.set_user_option('two', 'done')
        self.assertEqual('c2', c1._get_user_option('one'))

    def test_writes_are_serialized(self):
        c1 = self.config
        c2 = self.get_existing_config()
        before_writing = threading.Event()
        after_writing = threading.Event()
        writing_done = threading.Event()
        c1_orig = c1._write_config_file

        def c1_write_config_file():
            before_writing.set()
            c1_orig()
            after_writing.wait()
        c1._write_config_file = c1_write_config_file

        def c1_set_option():
            c1.set_user_option('one', 'c1')
            writing_done.set()
        t1 = threading.Thread(target=c1_set_option)
        self.addCleanup(t1.join)
        self.addCleanup(after_writing.set)
        t1.start()
        before_writing.wait()
        self.assertTrue(c1._lock.is_held)
        self.assertRaises(errors.LockContention, c2.set_user_option, 'one', 'c2')
        self.assertEqual('c1', c1.get_user_option('one'))
        after_writing.set()
        writing_done.wait()
        c2.set_user_option('one', 'c2')
        self.assertEqual('c2', c2.get_user_option('one'))

    def test_read_while_writing(self):
        c1 = self.config
        ready_to_write = threading.Event()
        do_writing = threading.Event()
        writing_done = threading.Event()
        c1_orig = c1._write_config_file

        def c1_write_config_file():
            ready_to_write.set()
            do_writing.wait()
            c1_orig()
            writing_done.set()
        c1._write_config_file = c1_write_config_file

        def c1_set_option():
            c1.set_user_option('one', 'c1')
        t1 = threading.Thread(target=c1_set_option)
        self.addCleanup(t1.join)
        self.addCleanup(do_writing.set)
        t1.start()
        ready_to_write.wait()
        self.assertTrue(c1._lock.is_held)
        self.assertEqual('c1', c1.get_user_option('one'))
        c2 = self.get_existing_config()
        self.assertEqual('1', c2.get_user_option('one'))
        do_writing.set()
        writing_done.wait()
        c3 = self.get_existing_config()
        self.assertEqual('c1', c3.get_user_option('one'))