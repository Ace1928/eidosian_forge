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
def _handle_sts_session(self, queue, q):
    if not hasattr(self, 'sts_expiration'):
        sts_creds = self.generate_sts_session_token(self.transport_options.get('sts_role_arn'), self.transport_options.get('sts_token_timeout', 900))
        self.sts_expiration = sts_creds['Expiration']
        c = self._predefined_queue_clients[queue] = self.new_sqs_client(region=q.get('region', self.region), access_key_id=sts_creds['AccessKeyId'], secret_access_key=sts_creds['SecretAccessKey'], session_token=sts_creds['SessionToken'])
        return c
    elif self.sts_expiration.replace(tzinfo=None) < datetime.utcnow():
        sts_creds = self.generate_sts_session_token(self.transport_options.get('sts_role_arn'), self.transport_options.get('sts_token_timeout', 900))
        self.sts_expiration = sts_creds['Expiration']
        c = self._predefined_queue_clients[queue] = self.new_sqs_client(region=q.get('region', self.region), access_key_id=sts_creds['AccessKeyId'], secret_access_key=sts_creds['SecretAccessKey'], session_token=sts_creds['SessionToken'])
        return c
    else:
        return self._predefined_queue_clients[queue]