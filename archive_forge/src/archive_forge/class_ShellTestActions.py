import os
import sys
import tempfile
from unittest import mock
import uuid
import fixtures
import io
from keystoneauth1 import fixture as keystone_fixture
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from requests_mock.contrib import fixture as rm_fixture
import testscenarios
import testtools
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
import heatclient.shell
from heatclient.tests.unit import fakes
import heatclient.v1.shell
class ShellTestActions(ShellBase):

    def setUp(self):
        super(ShellTestActions, self).setUp()
        self.set_fake_env(FAKE_ENV_KEYSTONE_V2)

    def test_stack_cancel_update(self):
        self.register_keystone_auth_fixture()
        expected_data = {'cancel_update': None}
        self.mock_request_post('/stacks/teststack2/actions', 'The request is accepted for processing.', data=expected_data, status_code=202)
        self.mock_stack_list()
        update_text = self.shell('stack-cancel-update teststack2')
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(update_text, r)

    def test_stack_check(self):
        self.register_keystone_auth_fixture()
        expected_data = {'check': None}
        self.mock_request_post('/stacks/teststack2/actions', 'The request is accepted for processing.', data=expected_data, status_code=202)
        self.mock_stack_list()
        check_text = self.shell('action-check teststack2')
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(check_text, r)

    def test_stack_suspend(self):
        self.register_keystone_auth_fixture()
        expected_data = {'suspend': None}
        self.mock_request_post('/stacks/teststack2/actions', 'The request is accepted for processing.', data=expected_data, status_code=202)
        self.mock_stack_list()
        suspend_text = self.shell('action-suspend teststack2')
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(suspend_text, r)

    def test_stack_resume(self):
        self.register_keystone_auth_fixture()
        expected_data = {'resume': None}
        self.mock_request_post('/stacks/teststack2/actions', 'The request is accepted for processing.', data=expected_data, status_code=202)
        self.mock_stack_list()
        resume_text = self.shell('action-resume teststack2')
        required = ['stack_name', 'id', 'teststack2', '1']
        for r in required:
            self.assertRegex(resume_text, r)