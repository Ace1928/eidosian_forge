from keystoneauth1.identity import generic
from keystoneauth1 import session as keystone_session
from unittest.mock import Mock
from designateclient.tests import v2
from designateclient.v2.client import Client
def test_timeout_override_session_timeout(self):
    session = create_session(timeout=10)
    self.assertEqual(10, session.timeout)
    client = Client(session=session, timeout=2)
    self.assertEqual(2, client.session.timeout)
    self._call_request_and_check_timeout(client, 2)