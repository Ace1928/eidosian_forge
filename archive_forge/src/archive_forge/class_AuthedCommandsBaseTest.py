import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
class AuthedCommandsBaseTest(testtools.TestCase):

    def setUp(self):
        super(AuthedCommandsBaseTest, self).setUp()
        self.orig_sys_exit = sys.exit
        sys.exit = mock.Mock(return_value=None)
        self.orig_sys_argv = sys.argv
        sys.argv = ['fakecmd']

    def tearDown(self):
        super(AuthedCommandsBaseTest, self).tearDown()
        sys.exit = self.orig_sys_exit
        self.orig_sys_argv = sys.argv

    def test___init__(self):
        parser = common.CliOptions().create_optparser(False)
        common.AuthedCommandsBase.debug = True
        dbaas = mock.Mock()
        dbaas.authenticate = mock.Mock(return_value=None)
        dbaas.client = mock.Mock()
        dbaas.client.auth_token = mock.Mock()
        dbaas.client.service_url = mock.Mock()
        dbaas.client.authenticate_with_token = mock.Mock()
        common.AuthedCommandsBase._get_client = mock.Mock(return_value=dbaas)
        common.AuthedCommandsBase(parser)