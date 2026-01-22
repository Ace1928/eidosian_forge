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
def _poll_messages(self, timeout):
    """Consume messages, callbacks and return list of messages"""
    msglist = self.consumer.consume(self.max_poll_records, timeout)
    if len(self.assignment_dict) == 0 or len(msglist) == 0:
        raise ConsumerTimeout()
    messages = []
    for message in msglist:
        if message is None:
            break
        a = self.find_assignment(message.topic(), message.partition())
        if a is None:
            LOG.warning('Message for %s received on unassigned partition %d', message.topic(), message.partition())
        else:
            messages.append(message.value())
    if not self.use_auto_commit:
        self.consumer.commit(asynchronous=False)
    return messages