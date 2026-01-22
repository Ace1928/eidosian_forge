import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
class RedisSsl(Redis):
    """
    Redis SSL class offering connection to the broker over SSL.
    This does not currently support SSL settings through the url, only through
    the broker_use_ssl celery configuration.
    """

    def __init__(self, broker_url, *args, **kwargs):
        if 'broker_use_ssl' not in kwargs:
            raise ValueError('rediss broker requires broker_use_ssl')
        self.broker_use_ssl = kwargs.get('broker_use_ssl', {})
        super().__init__(broker_url, *args, **kwargs)

    def _get_redis_client_args(self):
        client_args = super()._get_redis_client_args()
        client_args['ssl'] = True
        if isinstance(self.broker_use_ssl, dict):
            client_args.update(self.broker_use_ssl)
        return client_args