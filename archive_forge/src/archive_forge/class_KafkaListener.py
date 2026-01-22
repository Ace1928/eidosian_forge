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
class KafkaListener(base.PollStyleListener):

    def __init__(self, conn):
        super(KafkaListener, self).__init__()
        self._stopped = eventletutils.Event()
        self.conn = conn
        self.incoming_queue = []
        self.poll(5)

    @base.batch_poll_helper
    def poll(self, timeout=None):
        while not self._stopped.is_set():
            if self.incoming_queue:
                return self.incoming_queue.pop(0)
            try:
                messages = self.conn.consume(timeout=timeout) or []
                for message in messages:
                    msg = OsloKafkaMessage(*unpack_message(message))
                    self.incoming_queue.append(msg)
            except driver_common.Timeout:
                return None

    def stop(self):
        self._stopped.set()
        self.conn.stop_consuming()

    def cleanup(self):
        self.conn.close()