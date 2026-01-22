from unittest import mock
import testtools
from aodhclient.v2 import alarm_history
class AlarmHistoryManagerTest(testtools.TestCase):

    def setUp(self):
        super(AlarmHistoryManagerTest, self).setUp()
        self.client = mock.Mock()

    @mock.patch.object(alarm_history.AlarmHistoryManager, '_get')
    def test_get(self, mock_ahm):
        ahm = alarm_history.AlarmHistoryManager(self.client)
        ahm.get('01919bbd-8b0e-451c-be28-abe250ae9b1b')
        mock_ahm.assert_called_with('v2/alarms/01919bbd-8b0e-451c-be28-abe250ae9b1b/history')

    @mock.patch.object(alarm_history.AlarmHistoryManager, '_post')
    def test_search(self, mock_ahm):
        ahm = alarm_history.AlarmHistoryManager(self.client)
        q = '{"and": [{"=": {"type": "gnocchi_resources_threshold"}}, {"=": {"alarm_id": "87bacbcb-a09c-4cb9-86d0-ad410dd8ad98"}}]}'
        ahm.search(q)
        expected_called_data = '{"filter": "{\\"and\\": [{\\"=\\": {\\"type\\": \\"gnocchi_resources_threshold\\"}}, {\\"=\\": {\\"alarm_id\\": \\"87bacbcb-a09c-4cb9-86d0-ad410dd8ad98\\"}}]}"}'
        mock_ahm.assert_called_with('v2/query/alarms/history', data=expected_called_data, headers={'Content-Type': 'application/json'})