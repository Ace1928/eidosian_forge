from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
class OpenstackCommandTestCase(tests.TestCase):

    def setUp(self):
        super(OpenstackCommandTestCase, self).setUp()

    @testtools.skip('Have no idea how to test super')
    def test_run(self):
        pass

    @testtools.skip('Unskip it when get_data will do smthg')
    def test_get_data(self):
        pass

    @testtools.skip('Unskip it when get_data will do smthg')
    def test_take_action(self):
        pass