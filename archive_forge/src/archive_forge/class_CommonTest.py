import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
class CommonTest(testtools.TestCase):

    def setUp(self):
        super(CommonTest, self).setUp()
        self.orig_sys_exit = sys.exit
        sys.exit = mock.Mock(return_value=None)

    def tearDown(self):
        super(CommonTest, self).tearDown()
        sys.exit = self.orig_sys_exit

    def test_methods_of(self):

        class DummyClass(object):

            def dummyMethod(self):
                print('just for test')
        obj = DummyClass()
        result = common.methods_of(obj)
        self.assertEqual(1, len(result))
        method = result['dummyMethod']
        self.assertIsNotNone(method)

    def test_check_for_exceptions(self):
        status = [400, 422, 500]
        for s in status:
            resp = mock.Mock()
            resp.status = s
            self.assertRaises(Exception, common.check_for_exceptions, resp, 'body')
        resp = mock.Mock()
        resp.status_code = 200
        common.check_for_exceptions(resp, 'body')

    def test_print_actions(self):
        cmd = 'test-cmd'
        actions = {'test': 'test action', 'help': 'help action'}
        common.print_actions(cmd, actions)
        pass

    def test_print_commands(self):
        commands = {'cmd-1': 'cmd 1', 'cmd-2': 'cmd 2'}
        common.print_commands(commands)
        pass

    def test_limit_url(self):
        url = 'test-url'
        limit = None
        marker = None
        self.assertEqual(url, common.limit_url(url))
        limit = 'test-limit'
        marker = 'test-marker'
        expected = 'test-url?marker=test-marker&limit=test-limit'
        self.assertEqual(expected, common.limit_url(url, limit=limit, marker=marker))