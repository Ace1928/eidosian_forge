import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
@ddt.ddt
class LogInsightDriverTestCase(test.TestCase):
    BASE_ID = '8d28af1e-acc0-498c-9890-6908e33eff5f'

    def setUp(self):
        super(LogInsightDriverTestCase, self).setUp()
        self._client = mock.Mock(spec=loginsight.LogInsightClient)
        self._project = 'cinder'
        self._service = 'osapi_volume'
        self._host = 'ubuntu'
        with mock.patch.object(loginsight, 'LogInsightClient', return_value=self._client):
            self._driver = loginsight.LogInsightDriver('loginsight://username:password@host', project=self._project, service=self._service, host=self._host)

    @mock.patch.object(loginsight, 'LogInsightClient')
    def test_init(self, client_class):
        client = mock.Mock()
        client_class.return_value = client
        loginsight.LogInsightDriver('loginsight://username:password@host')
        client_class.assert_called_once_with('host', 'username', 'password')
        client.login.assert_called_once_with()

    @ddt.data('loginsight://username@host', 'loginsight://username:p@ssword@host', 'loginsight://us:rname:password@host')
    def test_init_with_invalid_connection_string(self, conn_str):
        self.assertRaises(ValueError, loginsight.LogInsightDriver, conn_str)

    @mock.patch.object(loginsight, 'LogInsightClient')
    def test_init_with_special_chars_in_conn_str(self, client_class):
        client = mock.Mock()
        client_class.return_value = client
        loginsight.LogInsightDriver('loginsight://username:p%40ssword@host')
        client_class.assert_called_once_with('host', 'username', 'p@ssword')
        client.login.assert_called_once_with()

    def test_get_name(self):
        self.assertEqual('loginsight', self._driver.get_name())

    def _create_trace(self, name, timestamp, parent_id='8d28af1e-acc0-498c-9890-6908e33eff5f', base_id=BASE_ID, trace_id='e465db5c-9672-45a1-b90b-da918f30aef6'):
        return {'parent_id': parent_id, 'name': name, 'base_id': base_id, 'trace_id': trace_id, 'timestamp': timestamp, 'info': {'host': self._host}}

    def _create_start_trace(self):
        return self._create_trace('wsgi-start', '2016-10-04t11:50:21.902303')

    def _create_stop_trace(self):
        return self._create_trace('wsgi-stop', '2016-10-04t11:50:30.123456')

    @mock.patch('json.dumps')
    def test_notify(self, dumps):
        json_str = mock.sentinel.json_str
        dumps.return_value = json_str
        trace = self._create_stop_trace()
        self._driver.notify(trace)
        trace['project'] = self._project
        trace['service'] = self._service
        exp_event = {'text': 'OSProfiler trace', 'fields': [{'name': 'base_id', 'content': trace['base_id']}, {'name': 'trace_id', 'content': trace['trace_id']}, {'name': 'project', 'content': trace['project']}, {'name': 'service', 'content': trace['service']}, {'name': 'name', 'content': trace['name']}, {'name': 'trace', 'content': json_str}]}
        self._client.send_event.assert_called_once_with(exp_event)

    @mock.patch.object(loginsight.LogInsightDriver, '_append_results')
    @mock.patch.object(loginsight.LogInsightDriver, '_parse_results')
    def test_get_report(self, parse_results, append_results):
        start_trace = self._create_start_trace()
        start_trace['project'] = self._project
        start_trace['service'] = self._service
        stop_trace = self._create_stop_trace()
        stop_trace['project'] = self._project
        stop_trace['service'] = self._service
        resp = {'events': [{'text': 'OSProfiler trace', 'fields': [{'name': 'trace', 'content': json.dumps(start_trace)}]}, {'text': 'OSProfiler trace', 'fields': [{'name': 'trace', 'content': json.dumps(stop_trace)}]}]}
        self._client.query_events = mock.Mock(return_value=resp)
        self._driver.get_report(self.BASE_ID)
        self._client.query_events.assert_called_once_with({'base_id': self.BASE_ID})
        append_results.assert_has_calls([mock.call(start_trace['trace_id'], start_trace['parent_id'], start_trace['name'], start_trace['project'], start_trace['service'], start_trace['info']['host'], start_trace['timestamp'], start_trace), mock.call(stop_trace['trace_id'], stop_trace['parent_id'], stop_trace['name'], stop_trace['project'], stop_trace['service'], stop_trace['info']['host'], stop_trace['timestamp'], stop_trace)])
        parse_results.assert_called_once_with()