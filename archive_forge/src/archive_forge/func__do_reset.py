import socket
import threading
from kombu.common import ignore_errors
from kombu.utils.encoding import safe_str
from celery.utils.collections import AttributeDict
from celery.utils.functional import pass1
from celery.utils.log import get_logger
from . import control
def _do_reset(self, c, connection):
    self._close_channel(c)
    self.node.channel = connection.channel()
    self.consumer = self.node.listen(callback=self.on_message)
    self.consumer.consume()