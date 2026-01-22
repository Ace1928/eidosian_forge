import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib, urlparse, parse_qsl
from libcloud.common.types import MalformedResponseError
from libcloud.common.cloudstack import CloudStackConnection
class CloudStackMockDriver:
    host = 'nonexistent.'
    path = '/path'
    async_poll_frequency = 0
    name = 'fake'
    async_delay = 0