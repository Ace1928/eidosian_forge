import os
import os.path
import stat
import unittest
from fixtures import MockPatch, TempDir
from testtools import TestCase
from lazr.restfulclient.authorize.oauth import (
def test_useful_distro_name(self):
    self.useFixture(MockPatch('distro.name', return_value='Fooix'))
    self.useFixture(MockPatch('platform.system', return_value='FooOS'))
    self.useFixture(MockPatch('socket.gethostname', return_value='foo'))
    consumer = SystemWideConsumer('app name')
    self.assertEqual(consumer.key, 'System-wide: Fooix (foo)')