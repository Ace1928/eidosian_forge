import os
from unittest import mock
import fixtures
from keystoneauth1 import session
from testtools import matchers
import openstack.config
from openstack import connection
from openstack import proxy
from openstack import service_description
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack.tests.unit.fake import fake_service
def test_session_provided(self):
    mock_session = mock.Mock(spec=session.Session)
    mock_session.auth = mock.Mock()
    mock_session.auth.auth_url = 'https://auth.example.com'
    conn = connection.Connection(session=mock_session, cert='cert')
    self.assertEqual(mock_session, conn.session)
    self.assertEqual('auth.example.com', conn.config.name)