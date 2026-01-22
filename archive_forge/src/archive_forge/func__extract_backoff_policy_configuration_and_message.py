from SQS when you have short-running tasks (or a large number of workers).
from __future__ import annotations
import base64
import socket
import string
import uuid
from datetime import datetime
from queue import Empty
from botocore.client import Config
from botocore.exceptions import ClientError
from vine import ensure_promise, promise, transform
from kombu.asynchronous import get_event_loop
from kombu.asynchronous.aws.ext import boto3, exceptions
from kombu.asynchronous.aws.sqs.connection import AsyncSQSConnection
from kombu.asynchronous.aws.sqs.message import AsyncMessage
from kombu.log import get_logger
from kombu.utils import scheduling
from kombu.utils.encoding import bytes_to_str, safe_str
from kombu.utils.json import dumps, loads
from kombu.utils.objects import cached_property
from . import virtual
def _extract_backoff_policy_configuration_and_message(self, delivery_tag):
    try:
        message = self._delivered[delivery_tag]
        routing_key = message.delivery_info['routing_key']
    except KeyError:
        return (None, None, None, None)
    if not routing_key or not message:
        return (None, None, None, None)
    queue_config = self.channel.predefined_queues.get(routing_key, {})
    backoff_tasks = queue_config.get('backoff_tasks')
    backoff_policy = queue_config.get('backoff_policy')
    return (routing_key, message, backoff_tasks, backoff_policy)