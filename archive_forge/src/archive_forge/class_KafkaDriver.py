import logging
import threading
import confluent_kafka
from confluent_kafka import KafkaException
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
from oslo_utils import importutils
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers.kafka_driver import kafka_options
class KafkaDriver(base.BaseDriver):
    """Kafka Driver

    See :doc:`kafka` for details.
    """

    def __init__(self, conf, url, default_exchange=None, allowed_remote_exmods=None):
        conf = kafka_options.register_opts(conf, url)
        super(KafkaDriver, self).__init__(conf, url, default_exchange, allowed_remote_exmods)
        self.listeners = []
        self.virtual_host = url.virtual_host
        self.pconn = ProducerConnection(conf, url)

    def cleanup(self):
        self.pconn.close()
        for c in self.listeners:
            c.close()
        self.listeners = []
        LOG.info('Kafka messaging driver shutdown')

    def send(self, target, ctxt, message, wait_for_reply=None, timeout=None, call_monitor_timeout=None, retry=None, transport_options=None):
        raise NotImplementedError('The RPC implementation for Kafka is not implemented')

    def send_notification(self, target, ctxt, message, version, retry=None):
        """Send notification to Kafka brokers

        :param target: Message destination target
        :type target: oslo_messaging.Target
        :param ctxt: Message context
        :type ctxt: dict
        :param message: Message payload to pass
        :type message: dict
        :param version: Messaging API version (currently not used)
        :type version: str
        :param call_monitor_timeout: Maximum time the client will wait for the
            call to complete before or receive a message heartbeat indicating
            the remote side is still executing.
        :type call_monitor_timeout: float
        :param retry: an optional default kafka consumer retries configuration
                      None means to retry forever
                      0 means no retry
                      N means N retries
        :type retry: int
        """
        self.pconn.notify_send(target_to_topic(target, vhost=self.virtual_host), ctxt, message, retry)

    def listen(self, target, batch_size, batch_timeout):
        raise NotImplementedError('The RPC implementation for Kafka is not implemented')

    def listen_for_notifications(self, targets_and_priorities, pool, batch_size, batch_timeout):
        """Listen to a specified list of targets on Kafka brokers

        :param targets_and_priorities: List of pairs (target, priority)
                                       priority is not used for kafka driver
                                       target.exchange_target.topic is used as
                                       a kafka topic
        :type targets_and_priorities: list
        :param pool: consumer group of Kafka consumers
        :type pool: string
        """
        conn = ConsumerConnection(self.conf, self._url)
        topics = []
        for target, priority in targets_and_priorities:
            topics.append(target_to_topic(target, priority))
        conn.declare_topic_consumer(topics, pool)
        listener = KafkaListener(conn)
        return base.PollStyleListenerAdapter(listener, batch_size, batch_timeout)