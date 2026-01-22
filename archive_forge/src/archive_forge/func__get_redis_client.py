import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
def _get_redis_client(self, broker_options):
    connection_kwargs = {'password': self.password, 'sentinel_kwargs': broker_options.get('sentinel_kwargs')}
    sentinel = redis.sentinel.Sentinel([(self.host, self.port)], **connection_kwargs)
    redis_client = sentinel.master_for(self.master_name)
    return redis_client