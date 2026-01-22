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
def apply_backoff_policy(self, routing_key, delivery_tag, backoff_policy, backoff_tasks):
    queue_url = self.channel._queue_cache[routing_key]
    task_name, number_of_retries = self.extract_task_name_and_number_of_retries(delivery_tag)
    if not task_name or not number_of_retries:
        return None
    policy_value = backoff_policy.get(number_of_retries)
    if task_name in backoff_tasks and policy_value is not None:
        c = self.channel.sqs(routing_key)
        c.change_message_visibility(QueueUrl=queue_url, ReceiptHandle=delivery_tag, VisibilityTimeout=policy_value)