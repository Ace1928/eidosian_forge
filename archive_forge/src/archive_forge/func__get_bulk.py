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
def _get_bulk(self, queue, max_if_unlimited=SQS_MAX_MESSAGES, callback=None):
    """Try to retrieve multiple messages off ``queue``.

        Where :meth:`_get` returns a single Payload object, this method
        returns a list of Payload objects.  The number of objects returned
        is determined by the total number of messages available in the queue
        and the number of messages the QoS object allows (based on the
        prefetch_count).

        Note:
        ----
            Ignores QoS limits so caller is responsible for checking
            that we are allowed to consume at least one message from the
            queue.  get_bulk will then ask QoS for an estimate of
            the number of extra messages that we can consume.

        Arguments:
        ---------
            queue (str): The queue name to pull from.

        Returns
        -------
            List[Message]
        """
    max_count = self._get_message_estimate()
    if max_count:
        q_url = self._new_queue(queue)
        resp = self.sqs(queue=queue).receive_message(QueueUrl=q_url, MaxNumberOfMessages=max_count, WaitTimeSeconds=self.wait_time_seconds)
        if resp.get('Messages'):
            for m in resp['Messages']:
                m['Body'] = AsyncMessage(body=m['Body']).decode()
            for msg in self._messages_to_python(resp['Messages'], queue):
                self.connection._deliver(msg, queue)
            return
    raise Empty()