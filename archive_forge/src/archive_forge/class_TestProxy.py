import socket
from unittest import mock
from taskflow.engines.worker_based import proxy
from taskflow import test
from taskflow.utils import threading_utils
class TestProxy(test.MockTestCase):

    def setUp(self):
        super(TestProxy, self).setUp()
        self.topic = 'test-topic'
        self.broker_url = 'test-url'
        self.exchange = 'test-exchange'
        self.timeout = 5
        self.de_period = proxy.DRAIN_EVENTS_PERIOD
        self.conn_mock, self.conn_inst_mock = self.patchClass(proxy.kombu, 'Connection')
        self.exchange_mock, self.exchange_inst_mock = self.patchClass(proxy.kombu, 'Exchange')
        self.queue_mock, self.queue_inst_mock = self.patchClass(proxy.kombu, 'Queue')
        self.producer_mock, self.producer_inst_mock = self.patchClass(proxy.kombu, 'Producer')

        def _ensure(obj, func, *args, **kwargs):
            return func
        self.conn_inst_mock.drain_events.side_effect = [socket.timeout, socket.timeout, KeyboardInterrupt]
        self.conn_inst_mock.ensure = mock.MagicMock(side_effect=_ensure)
        self.connections_mock = self.patch('taskflow.engines.worker_based.proxy.kombu.connections', attach_as='connections')
        self.connections_mock.__getitem__().acquire().__enter__.return_value = self.conn_inst_mock
        self.conn_inst_mock.Producer.return_value.__enter__ = mock.MagicMock()
        self.conn_inst_mock.Producer.return_value.__exit__ = mock.MagicMock()
        self.conn_inst_mock.Consumer.return_value.__enter__ = mock.MagicMock()
        self.conn_inst_mock.Consumer.return_value.__exit__ = mock.MagicMock()
        self.on_wait_mock = mock.MagicMock(name='on_wait')
        self.master_mock.attach_mock(self.on_wait_mock, 'on_wait')
        self.resetMasterMock()

    def _queue_name(self, topic):
        return '%s_%s' % (self.exchange, topic)

    def proxy_start_calls(self, calls, exc_type=mock.ANY):
        return [mock.call.Queue(name=self._queue_name(self.topic), exchange=self.exchange_inst_mock, routing_key=self.topic, durable=False, auto_delete=True, channel=self.conn_inst_mock), mock.call.connection.Consumer(queues=self.queue_inst_mock, callbacks=[mock.ANY]), mock.call.connection.Consumer().__enter__(), mock.call.connection.ensure(mock.ANY, mock.ANY, interval_start=mock.ANY, interval_max=mock.ANY, max_retries=mock.ANY, interval_step=mock.ANY, errback=mock.ANY)] + calls + [mock.call.connection.Consumer().__exit__(exc_type, mock.ANY, mock.ANY)]

    def proxy_publish_calls(self, calls, routing_key, exc_type=mock.ANY):
        return [mock.call.connection.Producer(), mock.call.connection.Producer().__enter__(), mock.call.connection.ensure(mock.ANY, mock.ANY, interval_start=mock.ANY, interval_max=mock.ANY, max_retries=mock.ANY, interval_step=mock.ANY, errback=mock.ANY), mock.call.Queue(name=self._queue_name(routing_key), routing_key=routing_key, exchange=self.exchange_inst_mock, durable=False, auto_delete=True, channel=None)] + calls + [mock.call.connection.Producer().__exit__(exc_type, mock.ANY, mock.ANY)]

    def proxy(self, reset_master_mock=False, **kwargs):
        proxy_kwargs = dict(topic=self.topic, exchange=self.exchange, url=self.broker_url, type_handlers={})
        proxy_kwargs.update(kwargs)
        p = proxy.Proxy(**proxy_kwargs)
        if reset_master_mock:
            self.resetMasterMock()
        return p

    def test_creation(self):
        self.proxy()
        master_mock_calls = [mock.call.Connection(self.broker_url, transport=None, transport_options=None), mock.call.Exchange(name=self.exchange, durable=False, auto_delete=True)]
        self.assertEqual(master_mock_calls, self.master_mock.mock_calls)

    def test_creation_custom(self):
        transport_opts = {'context': 'context'}
        self.proxy(transport='memory', transport_options=transport_opts)
        master_mock_calls = [mock.call.Connection(self.broker_url, transport='memory', transport_options=transport_opts), mock.call.Exchange(name=self.exchange, durable=False, auto_delete=True)]
        self.assertEqual(master_mock_calls, self.master_mock.mock_calls)

    def test_publish(self):
        msg_mock = mock.MagicMock()
        msg_data = 'msg-data'
        msg_mock.to_dict.return_value = msg_data
        routing_key = 'routing-key'
        task_uuid = 'task-uuid'
        p = self.proxy(reset_master_mock=True)
        p.publish(msg_mock, routing_key, correlation_id=task_uuid)
        mock_producer = mock.call.connection.Producer()
        master_mock_calls = self.proxy_publish_calls([mock_producer.__enter__().publish(body=msg_data, routing_key=routing_key, exchange=self.exchange_inst_mock, correlation_id=task_uuid, declare=[self.queue_inst_mock], type=msg_mock.TYPE, reply_to=None)], routing_key)
        self.master_mock.assert_has_calls(master_mock_calls)

    def test_start(self):
        try:
            self.proxy(reset_master_mock=True).start()
        except KeyboardInterrupt:
            pass
        master_calls = self.proxy_start_calls([mock.call.connection.drain_events(timeout=self.de_period), mock.call.connection.drain_events(timeout=self.de_period), mock.call.connection.drain_events(timeout=self.de_period)], exc_type=KeyboardInterrupt)
        self.master_mock.assert_has_calls(master_calls)

    def test_start_with_on_wait(self):
        try:
            self.proxy(reset_master_mock=True, on_wait=self.on_wait_mock).start()
        except KeyboardInterrupt:
            pass
        master_calls = self.proxy_start_calls([mock.call.connection.drain_events(timeout=self.de_period), mock.call.on_wait(), mock.call.connection.drain_events(timeout=self.de_period), mock.call.on_wait(), mock.call.connection.drain_events(timeout=self.de_period)], exc_type=KeyboardInterrupt)
        self.master_mock.assert_has_calls(master_calls)

    def test_start_with_on_wait_raises(self):
        self.on_wait_mock.side_effect = RuntimeError('Woot!')
        try:
            self.proxy(reset_master_mock=True, on_wait=self.on_wait_mock).start()
        except KeyboardInterrupt:
            pass
        master_calls = self.proxy_start_calls([mock.call.connection.drain_events(timeout=self.de_period), mock.call.on_wait()], exc_type=RuntimeError)
        self.master_mock.assert_has_calls(master_calls)

    def test_stop(self):
        self.conn_inst_mock.drain_events.side_effect = socket.timeout
        pr = self.proxy(reset_master_mock=True)
        self.assertFalse(pr.is_running)
        t = threading_utils.daemon_thread(pr.start)
        t.start()
        pr.wait()
        self.assertTrue(pr.is_running)
        pr.stop()
        t.join()
        self.assertFalse(pr.is_running)