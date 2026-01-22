from unittest import mock
from neutron_lib.callbacks import events
from oslotest import base
class DataStoreEventPayloadTestCase(base.BaseTestCase):

    def test_states(self):
        e = events.DBEventPayload(mock.ANY, states=['s1'])
        self.assertEqual(['s1'], e.states)

    def test_desired_state(self):
        desired_state = {'k': object()}
        e = events.DBEventPayload(mock.ANY, desired_state=desired_state)
        self.assertEqual(desired_state, e.desired_state)
        desired_state['a'] = 'A'
        self.assertEqual(desired_state, e.desired_state)

    def test_is_not_persisted(self):
        e = events.DBEventPayload(mock.ANY, states=['s1'])
        self.assertFalse(e.is_persisted)
        e = events.DBEventPayload(mock.ANY, resource_id='1a')
        self.assertFalse(e.is_persisted)

    def test_is_persisted(self):
        e = events.DBEventPayload(mock.ANY, states=['s1'], resource_id='1a')
        self.assertTrue(e.is_persisted)

    def test_is_not_to_be_committed(self):
        e = events.DBEventPayload(mock.ANY, states=['s1'], resource_id='1a')
        self.assertFalse(e.is_to_be_committed)

    def test_is_to_be_committed(self):
        e = events.DBEventPayload(mock.ANY, states=[mock.ANY], resource_id='1a', desired_state=object())
        self.assertTrue(e.is_to_be_committed)

    def test_latest_state_with_desired_state(self):
        desired_state = object()
        e = events.DBEventPayload(mock.ANY, states=[object()], desired_state=desired_state)
        self.assertEqual(desired_state, e.latest_state)