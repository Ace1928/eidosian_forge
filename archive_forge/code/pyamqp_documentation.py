from __future__ import annotations
import amqp
from kombu.utils.amq_manager import get_manager
from kombu.utils.text import version_string_as_tuple
from . import base
from .base import to_rabbitmq_queue_arguments
Close the AMQP broker connection.