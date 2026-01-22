import os
import os.path
import stat
import unittest
from fixtures import MockPatch, TempDir
from testtools import TestCase
from lazr.restfulclient.authorize.oauth import (
class TestSystemWideConsumer(TestCase):

    def test_useful_distro_name(self):
        self.useFixture(MockPatch('distro.name', return_value='Fooix'))
        self.useFixture(MockPatch('platform.system', return_value='FooOS'))
        self.useFixture(MockPatch('socket.gethostname', return_value='foo'))
        consumer = SystemWideConsumer('app name')
        self.assertEqual(consumer.key, 'System-wide: Fooix (foo)')

    def test_empty_distro_name(self):
        self.useFixture(MockPatch('distro.name', return_value=''))
        self.useFixture(MockPatch('platform.system', return_value='BarOS'))
        self.useFixture(MockPatch('socket.gethostname', return_value='bar'))
        consumer = SystemWideConsumer('app name')
        self.assertEqual(consumer.key, 'System-wide: BarOS (bar)')

    def test_broken_distro_name(self):
        self.useFixture(MockPatch('distro.name', side_effect=Exception('Oh noes!')))
        self.useFixture(MockPatch('platform.system', return_value='BazOS'))
        self.useFixture(MockPatch('socket.gethostname', return_value='baz'))
        consumer = SystemWideConsumer('app name')
        self.assertEqual(consumer.key, 'System-wide: BazOS (baz)')