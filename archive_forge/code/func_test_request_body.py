from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
def test_request_body(self):
    e = events.EventPayload(mock.ANY, request_body={'k', 'v'})
    self.assertEqual({'k', 'v'}, e.request_body)