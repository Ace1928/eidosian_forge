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
def _messages_to_python(self, messages, queue):
    """Convert a list of SQS Message objects into Payloads.

        This method handles converting SQS Message objects into
        Payloads, and appropriately updating the queue depending on
        the 'ack' settings for that queue.

        Arguments:
        ---------
            messages (SQSMessage): A list of SQS Message objects.
            queue (str): Name representing the queue they came from.

        Returns
        -------
            List: A list of Payload objects
        """
    q_url = self._new_queue(queue)
    return [self._message_to_python(m, queue, q_url) for m in messages]